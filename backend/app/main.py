print("Jobayer Faisal Fahim")

print("I am learning Docker and PostgreSQL integration.")
print("This is part of the Emergency Response System project.")
print("I hope to deploy this application successfully.")
print("This will help in managing disaster response effectively.")
print("Thank you for reviewing my code changes.")
print("I am excited to see how Docker and PostgreSQL work together.")
print("This experience is enhancing my backend development skills.")
print("Looking forward to more projects like this in the future!")





# --- IGNORE ---

from fastapi import FastAPI

app = FastAPI(title="Emergency Response Backend")

@app.get("/api/v1/health")
def health_check():
    return {"status": "ok"}
