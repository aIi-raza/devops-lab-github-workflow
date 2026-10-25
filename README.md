# DevOps Lab: CI/CD Calculator

Angular frontend + Flask backend, deployed with GitHub Actions.

Live site: https://aii-raza.github.io/devops-lab-github-workflow/

## Structure
- `frontend/`  Angular app (deployed to GitHub Pages)
- `backend/`   Flask API (`/calculate`), runs locally
- `.github/workflows/ci.yml`      CI: checks Flask, runs backend tests, builds Angular
- `.github/workflows/deploy.yml`  CD: runs after CI passes, deploys to GitHub Pages

## Run the backend
```
cd backend
pip install -r requirements.txt
python app.py
```
Runs at http://127.0.0.1:5000

## Run the frontend
```
cd frontend
npm install
npm start
```
Opens at http://localhost:4200

## Demo changes
- TypeScript: `frontend/src/app/app.ts` -> change `version = 'v1'`
- HTML: `frontend/src/app/app.html` -> change text
- CSS: `frontend/src/app/app.css` -> change `background: #1e3a8a;`
