# Deploying the search service

The commands are for PowerShell 7, run from the root of a checkout of the
commit to deploy. Every command names the project, since gcloud's default
project is another one. A key is read from `.env`, written to a temporary file
without a line end, and never printed; a line end would become part of the key.

The service is reachable without Google's sign-in, as the inference service
is, so its address is taken to be public. `/health` without the key says only
that the service is up and which package it holds. Everything else needs the
shared secret in `X-Search-Key`, which the service checks before it reads a
byte of the body, and a body holds at most 8 KiB.

## Once per terminal

```powershell
$env:CLOUDSDK_PYTHON = "C:\Program Files (x86)\Google\Cloud SDK\google-cloud-sdk\platform\bundledpython\python.exe"
$project = "maktsprak-inference"
$region = "us-central1"
$runner = "maktsprak-search@$project.iam.gserviceaccount.com"
$image = "us-central1-docker.pkg.dev/$project/cloud-run-source-deploy/maktsprak-search"
$envFile = Resolve-Path .env
```

## The search alone, in one run

This run deploys the search without the answer step. In order, it:

1. turns on the APIs for Cloud Scheduler and budgets;
2. gives the service an account of its own that can read its two secrets and nothing else;
3. draws a new shared secret, keeps it in `.env` and adds it to Secret Manager as a new version;
4. builds the image;
5. deploys the image with that version, under the service's own account. The answer step is off, the Anthropic key and the ledger's volume are gone, and an instance takes at most eight requests at a time;
6. checks the deployment. Go on only if every check holds;
7. disables the old version of the secret and takes the default account's access to the service's secrets and bucket away;
8. asks `/health` every five minutes from Cloud Scheduler, so that a visitor rarely meets a cold start;
9. sets a budget on the project, with alerts by email;
10. keeps the two latest images in Artifact Registry and deletes the older ones.

Old preview deployments on Vercel hold the old secret, so from step 5 on the
service answers them with 401. Their `/api/fraga` then answers 502.

```powershell
# 1. The APIs.
gcloud services enable cloudscheduler.googleapis.com billingbudgets.googleapis.com --project $project

# 2. The service's own account, with its two secrets.
gcloud iam service-accounts create maktsprak-search --display-name "maktsprak-search" --project $project
foreach ($name in "openrouter-api-key", "search-shared-secret") {
  gcloud secrets add-iam-policy-binding $name --member="serviceAccount:$runner" --role="roles/secretmanager.secretAccessor" --project $project
}

# 3. A new shared secret, in .env and as a new version.
if (-not (Select-String -Path $envFile -Pattern '^SEARCH_SHARED_SECRET=' -Quiet)) { throw "SEARCH_SHARED_SECRET is not in .env" }
$bytes = [byte[]]::new(32); [Security.Cryptography.RandomNumberGenerator]::Fill($bytes)
$secret = [Convert]::ToHexString($bytes).ToLower()
(Get-Content $envFile) -replace '^SEARCH_SHARED_SECRET=.*$', "SEARCH_SHARED_SECRET=$secret" | Set-Content $envFile
$file = New-TemporaryFile
[IO.File]::WriteAllText($file, $secret)
$version = (gcloud secrets versions add search-shared-secret --data-file=$file --project $project --format="value(name)").Split("/")[-1]
Remove-Item $file
"search-shared-secret version $version holds $((gcloud secrets versions access $version --secret search-shared-secret --project $project).Length) characters"

# 4. The image.
$tag = git rev-parse --short HEAD
gcloud builds submit --project $project --region $region --config search_service/cloudbuild.yaml --substitutions "_TAG=$tag" .

# 5. The search alone, with the new version of the secret, under its own account.
gcloud run deploy maktsprak-search `
  --image "${image}:$tag" `
  --region $region --project $project `
  --service-account $runner `
  --execution-environment gen2 --cpu 1 --memory 4Gi --cpu-boost `
  --concurrency 8 --max-instances 2 --timeout 300 --allow-unauthenticated `
  --set-secrets "OPENROUTER_API_KEY=openrouter-api-key:1,SEARCH_SHARED_SECRET=search-shared-secret:$version" `
  --set-env-vars "FRAGA_ANSWER_STEP=off" `
  --clear-volume-mounts --clear-volumes

# 6. The checks.
$url = gcloud run services describe maktsprak-search --region $region --project $project --format="value(status.url)"
gcloud run services describe maktsprak-search --region $region --project $project --format="yaml(spec.template.spec.serviceAccountName,spec.template.spec.containerConcurrency,spec.template.spec.containers[0].env)"
$key = @{ "X-Search-Key" = $secret }
Invoke-RestMethod "$url/health"
Invoke-RestMethod "$url/health" -Headers $key | Select-Object status, answer_step, answer_report, selector, ledger, manifest_sha256
$body = '{"question":"Vad har sagts om vindkraft?","filters":{}}'
foreach ($check in @(@("search", $key, 200), @("answer", $key, 403), @("parties", $key, 403), @("search", @{}, 401))) {
  $code = (Invoke-WebRequest "$url/$($check[0])" -Method Post -Body $body -ContentType "application/json" -Headers $check[1] -SkipHttpErrorCheck).StatusCode
  "/$($check[0]): $code, expected $($check[2])"
}

# 7. The old secret, and the default account's access.
gcloud secrets versions disable 1 --secret search-shared-secret --project $project
$default = "$(gcloud projects describe $project --format='value(projectNumber)')-compute@developer.gserviceaccount.com"
foreach ($name in "openrouter-api-key", "search-shared-secret") {
  gcloud secrets remove-iam-policy-binding $name --member="serviceAccount:$default" --role="roles/secretmanager.secretAccessor" --project $project
}
gcloud storage buckets remove-iam-policy-binding gs://maktsprak-search-ledger --member="serviceAccount:$default" --role="roles/storage.objectUser" --project $project

# 8. /health every five minutes.
gcloud scheduler jobs create http maktsprak-search-warm --project $project --location $region --schedule "*/5 * * * *" --uri "$url/health" --http-method GET --attempt-deadline 60s --description "Keeps the search service warm"

# 9. A budget on the project, with alerts by email at half, nine tenths and all of it.
$account = (gcloud billing projects describe $project --format="value(billingAccountName)").Split("/")[-1]
gcloud billing budgets create --billing-account $account --display-name "maktsprak-inference" --budget-amount 10USD --filter-projects "projects/$project" --threshold-rule percent=0.5 --threshold-rule percent=0.9 --threshold-rule percent=1.0

# 10. The two latest images, and no older ones.
gcloud artifacts repositories set-cleanup-policies cloud-run-source-deploy --project $project --location $region --policy search_service/cleanup-policy.json --no-dry-run
```

Step 6 holds when:

- the environment shows `secretKeyRef` for the two secrets, never a value;
- the new secret has its new version, `FRAGA_ANSWER_STEP` is `off`, and there is no `ANTHROPIC_API_KEY`;
- the account is the service's own, and the concurrency is 8;
- `/health` without the key gives only `status` and `manifest_sha256`;
- `/health` with the key gives `answer_step` `off`, `answer_report` `missing`, `selector` `none` and `ledger` `none`;
- every request gives the code it is expected to.

The cleanup policy is scoped to `maktsprak-search`, so the inference service's
images in the same repository are not touched.

## The site's proxy

`web/src/app/api/fraga/route.ts` reads `SEARCH_SERVICE_URL` (the `$url` above)
and `SEARCH_SHARED_SECRET` (as in `.env`) from Vercel's environment. Both are
set for **Production only**. Preview has neither, so a preview's `/api/fraga`
answers 501 and never reaches the service.

In Vercel's dashboard, under the project's Settings and Environment Variables:

1. Delete `SEARCH_SERVICE_URL` and `SEARCH_SHARED_SECRET` from Preview.
2. Add both for Production. Copy the secret's value from `.env`.
3. Leave `FRAGA_ANSWER_STEP` unset, or `off`, in every environment.

A change to the variables takes effect with the next deployment, so the
production deployment of the launch, built after this, is the first to hold
them.

## A later deploy of a new image

The service keeps its account, secrets and settings when only the image
changes:

```powershell
$tag = git rev-parse --short HEAD
gcloud builds submit --project $project --region $region --config search_service/cloudbuild.yaml --substitutions "_TAG=$tag" .
gcloud run deploy maktsprak-search --image "${image}:$tag" --region $region --project $project
```

Before a version of a secret is disabled, the service is deployed on its
successor: an instance whose secret cannot be read does not start.

## Before the answer step is turned on

The answer step stays off until all of this holds:

1. **The launch report.** `data/reports/search/answer_test.json` is
   committed, and its launch rule holds. The image carries the report, and the
   service runs the answer step only when the report says it launches, with
   the selector the report says runs. Without the report, the switch alone
   does nothing.
2. **The hard cap.** The service's Anthropic key belongs to a workspace of its
   own in Anthropic's console, with a spending limit of 2 USD a month. That is
   the cap that holds if the service's own count is wrong. Put that key into
   `anthropic-api-key` as a new version.
3. **The ledger.** The service counts the month's spend in
   `gs://maktsprak-search-ledger/budget.json`. It writes the object only on the
   generation it read, so two instances count together. The guard checks the
   budget before each request to the model and counts each request as it
   returns. The month can therefore pass the budget only by the requests
   already sent when it is reached, at most one per instance. The account
   needs the bucket:

   ```powershell
   gcloud storage buckets add-iam-policy-binding gs://maktsprak-search-ledger --member="serviceAccount:$runner" --role="roles/storage.objectUser" --project $project
   gcloud secrets add-iam-policy-binding anthropic-api-key --member="serviceAccount:$runner" --role="roles/secretmanager.secretAccessor" --project $project
   $keyVersion = (gcloud secrets versions list anthropic-api-key --project $project --filter="state:ENABLED" --sort-by="~createTime" --limit 1 --format="value(name)").Split("/")[-1]
   gcloud run deploy maktsprak-search --region $region --project $project `
     --update-secrets "ANTHROPIC_API_KEY=anthropic-api-key:$keyVersion" `
     --update-env-vars "FRAGA_ANSWER_STEP=on,BUDGET_LEDGER=gs://maktsprak-search-ledger/budget.json"
   ```

   `/health` with the key then says `answer_step` `on`, `answer_report`
   `launch`, `selector` `claude` (or `baseline`, when the report has the
   baseline run) and `ledger` `gcs`.
4. **The switch in Vercel.** `FRAGA_ANSWER_STEP` is `on` for Production only,
   never in Preview, whose deployments are public.
5. **What one caller may spend.** One caller may ask 10 answers an hour on
   each instance (`answers_per_window` in `search/config.yaml`), and the month
   holds roughly 50 answers. A lower limit is a change of the settings. A
   daily share of the budget would change the registered rule that the
   baseline answers once the month's budget is spent, so it needs an addendum
   first. So would a lower ceiling than 8 000 output tokens, or no retry of a
   cut-off answer.

## What it costs

Cloud Run bills the service by request. A request to `/health` takes less
than the 100 ms Cloud Run bills at the least, so the scheduler's 8 640
requests a month hold the instance for about a quarter of an hour. A search
through the page holds it for one to two seconds, and up to 30 s when the
embedding API is slow. All of it is inside Cloud Run's free tier at this
page's traffic. The scheduler's job is one of the three free on a billing
account. The image holds the package and is about 4 GB; the two images kept
come to about 8 GB, which is about 0.75 USD a month past Artifact Registry's
free 0.5 GB. The budget's alerts report anything beyond that.
