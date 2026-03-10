# 1. Baseline Project Access
- Create a dedicated project corresponding to the Image-to-Product pipeline
- Permissions for viewing project resources and logs
- Service Account for the pipeline (Pipeline will run under the scope of the service account)

# 2. Services / APIs
- Cloud Storage (for image/object storage - Currently using open-source project MinIO in local)
- Secret Manager (for managing API keys, web scraper credentials, ElasticSearch database credentials, etc.)
- Cloud Run (for executing containerized pipeline services + APIs)
- Artifact Registry (for storing Docker images)
- Cloud Logging (for debugging and audit trails)
- Pub/Sub (for action based triggers i.e. requesting images for a specific project without manually executing the web scraper)
- Vertex AI (for running Image Classifier / Text models)
- Cloud Scheduler (for potential use later to request updating images because of age / lacking quality)

# 3. IAM Permissions (User)
- Cloud Run Developer (for running cloud services)
- Artifact Registry Reader/Writer (for pushing images to the models)
- Storage Object Admin (for access to specific cloud storage bucket(s) storing images / metadata)
- Secret Manager / Secret Accessor (for reading secrets i.e. API keys)
- Logs Viewer (for debugging purposes)
- Service Account User (permission to use runtime service account)
- Vertex AI User (for invoking hosted models / endpoints)

# 4. IAM roles (Pipeline Service Account)
- Read/Write to image storage bucket
- Read access to secrets (i.e. API keys)
- Write logs (testing and debugging)
- Call Vertex AI endpoints for models hosted there
- Publish to Pub/Sub (potential use later)