1. **Frontend (React):**
   We use **React** as the user interface, allowing doctors and healthcare professionals to **upload patient data** such as CT scans, X-rays, and medical reports. The platform also provides **visualizations**, including **Grad-CAM heatmaps**, to help interpret AI-generated predictions and insights.

2. **Backend (Spring Boot):**
   **Spring Boot** handles **secure authentication, request management, and business logic**. It serves as the **primary gateway**, managing communication between the frontend, database, and AI models. This ensures a smooth, secure, and centralized workflow.

3. **AI Model API (FastAPI):**
   We use **FastAPI** to **expose endpoints for our AI models**. Spring Boot communicates with FastAPI, allowing doctors to **interact seamlessly with the AI model** through a single integrated interface.

4. **Database (PostgreSQL):**
   **PostgreSQL** is used to **store both structured and unstructured data**. Structured data includes patient records, while unstructured data, like CT scans and X-rays, is stored using **BLOB (Binary Large Object) data types**.
