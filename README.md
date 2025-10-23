# Heart Tells

## What does Heart Tells do?
Making a hard decision? Put down two decisions and let your heart tell you.
Stages include negative, highly positive, positive, and neutral.
Use Chatgpt to help you go through the logic of the choices and explaining what your heart feeling by ECG signals respectively.
And help you to balance your mind and heart.

## Features
#### Login/Register and User Onboarding
    - platform: Web app (Progressive Web App support for mobile)
    - Authentication: 
        - Sign up/in with Google (OAuth 2.0)
        - Sign up/in with Apple (OAuth 2.0)
        - Sign up/in with email/password
    - Questionaaier (User management):
        - Background questionaire
        - ECG signal template questionaire
    - User management:
        - User profile
        - User settings
        - User authentication

#### User Flow
    - auth with google account or apple account
    - questionaires to understand brief background
    - record ECG signals and heart rate (5 2seconds segments)

#### Stage I
    - chat with chatgpt and leave two choices
    - sentiment analysis of the choices
    - logic of the choices
    - sentiment analysis of the ECG signals and heart rate
    - loca embeddings of the decisions and ECG signals
    - output the small summary of the decisions and ECG signals

#### Stage II
    - Able to advance make the path of the decisions to track progress and make decisions modularly.
    - output the small summary of the decisions path and confidence feeling of the decisions

## Tech Stack
    - auth: Google, Apple
    - Frontend: React, Typescript
    - Backend: FastAPI, PostgreSQL
    - NLP: ChatGPT, ECG signals, ECG embeddings, sentiment analysis
    - Deployment: GCP
    - CI/CD: Github Actions
    - Others: Docker, Kubernetes, Helm, Prometheus, Grafana (monitoring)
    - connect to google calendar or anki schedular

## user Flowchart
```mermaid

flowchart TD
    A[You] --> B[Login]
    A --> C[Register]
    C --> J[Authenticate with Google/Apple account]
    J --> K[Complete Background Questionaires]
    K --> L[Create ECG signal templates]
    L --> M[Build ECG embeddings]
    B --> D[What decisions you want to make?]
    D --> E[Measure the ECG signals of each decisions]
    E --> F[ECG signals mapping ECG embeddings to decisions]
    M --> F
    F --> G[ChatGPT with decisions and ECG embeddings and sentiment analysis]
    G -- Basic--> H[Stage I]
    G -- Advance --> I[Stage II]
```
## Architecture  (Check for better display)

```mermaid
flowchart LR
    
    subgraph Frontend["React/TypeScript web App"]
        A1["Login Page Google/Apple Auth"]
        A2["Chat Interface ChatGPT API"]
        A3["Dashboard ECG + Sentiment Summary"]
        A4[Decision Path Visualization if needed]
    end
    subgraph backend["FastAPI Backend on GCP"]
        B1["User Management (OAuth tokens)"]
        B2["ECG Data Endpoint"]
        B3["Decision & Sentiment API"]
        B4["Embedding Fusion Engine"]
        B5[PostgreSQL Database]
        B6["Monitoring + Logging (Prometheus/Grafana"]
        B1 --> B5
        B2 --> B5
        B3 --> B4 --> B5
    end
    
    subgraph Apple["Apple Ecosystem"]
        C1[Apple Watch ECG App]
        C2[HealthKit: HKElectrocardiogram]
        C3[iOS Companion App]
        C1 --> C2 --> C3 --> B2
    end
    
    subgraph CI/CD[CI/CD & Deployment]
        D1[Github Actions]
        D2[Docker Build]
        D3["Kubernetes (GKE)"]
        D4[Helm Charts]
        D5[Monitoring Stack]
        D1 --> D2 --> D3 --> D4 --> D5
    end
    A1 --> B1 
    A2 --> B3
    A3 --> B2
    A4 --> B5
    B2 -->|ECG Sync| C3
    B5 --> A3
```
## Connect to Apple Watch to get ECG signals (Check for better display)

```mermaid
flowchart TD
    
    subgraph WatchLayer ["Apple watch layer"]
        A1[ECG Sensor Measurement]
        A2[ECG App on Watch]
        A1 --> A2
    end
    
    subgraph HealthKitLayer ["HealthKit Layer (iphone)"]
        B1[HKElectrocardiogram Dta Type]
        B2[Health App Storage]
        A2 -->|Bluetooth Sync| B1 --> B2
    end
    
    subgraph AppLayer ["iOS App Layer"]
    C1[HKOberserverQuery -> detect new ECG]
    C2[SampleQuery -> fetch metadata]
    C3[HKelectrocardiogramQuery -> stream voltage samples]
    C4[Preprocessing: detrend, filter, normalize]
    C5{{Segmentation}}
    C6[Resampling to fixed N]
    C7[On-device Embedding Model e.q. 1D-CNN Core ML]
    C8[(New ECG segment of decision1)]
    C9[(New ECG segment of decision2)]
    C10[ECG signals map to embeddings]
    C11[Similarity Computation]
    C12[Store Results and Metadata and return to App]
    C1 --> C2 --> C3 --> C4 --> C5 --> C6 --> C7
    C7 --> C8
    C7 --> C9
    C8 --> C10 & C11
    C9 --> C10 & C11
    C10 & C11 --> C12
    end

```
## How to use it?
