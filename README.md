chronic kidney disease using xgboost

## Project Overview
This project consists of a **React frontend** and a **Streamlit backend** for a web application. The frontend is built using React and runs on Vite, while the backend uses Streamlit for machine learning-based predictions.

---

## Installation and Setup

### **Frontend (React)**
#### Prerequisites:
- Node.js installed (Download from [here](https://nodejs.org/))

#### **Steps to Run Frontend**
1. Navigate to the frontend directory:
   ```sh
   cd frontend
   ```
2. Install dependencies:
   ```sh
   npm install
   ```
3. Start the development server:
   ```sh
   npm run dev
   ```
4. The React app should now be running at `http://localhost:5173/` (or another available port).

---

### **Backend (Python - Streamlit)**
#### Prerequisites:
- Python 3 installed (Download from [here](https://www.python.org/downloads/))
- Virtual environment setup (recommended)

#### **Steps to Run Backend**
1. Navigate to the backend directory:
   ```sh
   cd backend
   ```
2. Install required Python dependencies:
   ```sh
   pip install -r requirements.txt
   ```
3. Run the Streamlit application:
   ```sh
   streamlit run app.py
   ```
4. The backend should now be accessible at `http://localhost:8501/`.

---

## **Project Structure**
```
/newsat
│-- frontend/  # React frontend
│   │-- src/
│   │-- public/
│   │-- package.json
│   │-- vite.config.js
│
│-- backend/  # Streamlit backend
│   │-- app.py
│   │-- models/
│   │-- requirements.txt
│
│-- README.md  # Project documentation
│-- .gitignore
```

---

## **Contributing**
If you wish to contribute:
1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes and commit (`git commit -m 'Added new feature'`).
4. Push to your fork (`git push origin feature-branch`).
5. Submit a Pull Request.

---

## **License**
This project is licensed under the MIT License.

---

For any issues, feel free to reach out or create an issue in the repository. 🚀
