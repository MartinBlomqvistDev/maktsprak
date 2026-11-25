# Deploying the search service

The commands are for PowerShell, run from the repository root. Every command
names the project, since gcloud's default project is another one. A key is
read from `.env`, written to a temporary file without a line end, and never
printed; a line end would become part of the key.

## Once per terminal

```powershell
$env:CLOUDSDK_PYTHON = "C:\Program Files (x86)\Google\Cloud SDK\google-cloud-sdk\platform\bundledpython\python.exe"
$project = "maktsprak-inference"
$region = "us-central1"
$runner = "796285283223-compute@developer.gserviceaccount.com"
function New-SecretFromEnv([string]$name, [string]$secret) {
  $line = Get-Content .env | Where-Object { $_ -match "^$name=" } | Select-Object -First 1
  $value = $line.Substring($name.Length + 1).Trim().Trim('"')
  $file = New-TemporaryFile
  [IO.File]::WriteAllText($file, $value)
  gcloud secrets create $secret --data-file=$file --replication-policy=automatic --project $project
  Remove-Item $file
  $stored = (gcloud secrets versions access 1 --secret $secret --project $project).Length
  "length in .env $($value.Length), stored $stored"
}
```

## Secrets and the budget's ledger, once

The selector's key is in Secret Manager already (`anthropic-api-key`). The
embedding API's key and the secret the site's proxy sends become secrets
beside it. The shared secret is drawn here and kept in `.env`, from which the
proxy's environment gets it too.

```powershell
New-SecretFromEnv "OPENROUTER_API_KEY" "openrouter-api-key"

$bytes = [byte[]]::new(32); [Security.Cryptography.RandomNumberGenerator]::Fill($bytes)
if (-not (Get-Content .env -Raw).EndsWith("`n")) { Add-Content .env "" }
Add-Content .env "SEARCH_SHARED_SECRET=$([Convert]::ToHexString($bytes).ToLower())"
New-SecretFromEnv "SEARCH_SHARED_SECRET" "search-shared-secret"

foreach ($secret in "openrouter-api-key", "search-shared-secret") {
  gcloud secrets add-iam-policy-binding $secret --member="serviceAccount:$runner" --role="roles/secretmanager.secretAccessor" --project $project
}
```

The budget guard keeps the month's spend on the language model in a file. A
file in the container would start the month again at every cold start, so it
lives on a bucket that the service mounts:

```powershell
gcloud storage buckets create gs://maktsprak-search-ledger --project $project --location $region --uniform-bucket-level-access
gcloud storage buckets add-iam-policy-binding gs://maktsprak-search-ledger --member="serviceAccount:$runner" --role="roles/storage.objectUser"
```

## Build and deploy

The image pins its package's revision and manifest hash
(`search_service/Dockerfile`), so the build fetches exactly that package and
fails if it does not check out. `.gcloudignore` keeps the upload to `search/`
and `search_service/`.

```powershell
$tag = git rev-parse --short HEAD
gcloud builds submit --project $project --region $region --config search_service/cloudbuild.yaml --substitutions "_TAG=$tag" .

gcloud run deploy maktsprak-search `
  --image "us-central1-docker.pkg.dev/$project/cloud-run-source-deploy/maktsprak-search:$tag" `
  --region $region --project $project `
  --execution-environment gen2 --cpu 1 --memory 4Gi --cpu-boost `
  --max-instances 2 --timeout 300 --allow-unauthenticated `
  --set-secrets "ANTHROPIC_API_KEY=anthropic-api-key:1,OPENROUTER_API_KEY=openrouter-api-key:1,SEARCH_SHARED_SECRET=search-shared-secret:1" `
  --add-volume "name=ledger,type=cloud-storage,bucket=maktsprak-search-ledger" `
  --add-volume-mount "volume=ledger,mount-path=/ledger"
```

The service is reachable without Google's sign-in, as the inference service
is, but answers a question only with the shared secret in `X-Search-Key`.

## Check

The environment must show `secretKeyRef` for the three secrets, never a value,
and `/health` must name the package's manifest hash and `"ledger": "file"`.

```powershell
gcloud run services describe maktsprak-search --region $region --project $project --format="yaml(spec.template.spec.containers[0].env)"
$url = gcloud run services describe maktsprak-search --region $region --project $project --format="value(status.url)"
Invoke-RestMethod "$url/health"
```

## The site's proxy

`web/src/app/api/fraga/route.ts` reads `SEARCH_SERVICE_URL` (the `$url` above)
and `SEARCH_SHARED_SECRET` (as in `.env`) from Vercel's environment. Set both
for the Preview environment first, and for Production once the preview
answers.

## What it costs

Cloud Run bills the service by request, inside the free tier at this page's
traffic, and the ledger's bucket holds one small file. The image holds the
package and is about 4 GB, more than Artifact Registry stores for free, so
keep only the image that runs.
