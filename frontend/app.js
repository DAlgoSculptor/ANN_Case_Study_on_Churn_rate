const form = document.getElementById("predictForm");
const resultEl = document.getElementById("result");
const jsonDumpEl = document.getElementById("jsonDump");
const apiBaseUrlEl = document.getElementById("apiBaseUrl");
const thresholdEl = document.getElementById("threshold");
const predictBtn = document.getElementById("predictBtn");
const fillExample2Btn = document.getElementById("fillExample2Btn");

function setResult(text, kind = "muted") {
  resultEl.className = `result ${kind}`;
  resultEl.textContent = text;
}

function formToCustomer(formEl) {
  const fd = new FormData(formEl);
  const obj = Object.fromEntries(fd.entries());

  return {
    CreditScore: Number(obj.CreditScore),
    Geography: String(obj.Geography),
    Gender: String(obj.Gender),
    Age: Number(obj.Age),
    Tenure: Number(obj.Tenure),
    Balance: Number(obj.Balance),
    NumOfProducts: Number(obj.NumOfProducts),
    HasCrCard: Number(obj.HasCrCard),
    IsActiveMember: Number(obj.IsActiveMember),
    EstimatedSalary: Number(obj.EstimatedSalary),
  };
}

function pretty(obj) {
  return JSON.stringify(obj, null, 2);
}

async function predict(customer, threshold) {
  const base = apiBaseUrlEl.value.replace(/\/+$/, "");
  const url = `${base}/predict`;

  const payload = { customer, threshold };
  const resp = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

  const data = await resp.json();
  if (!resp.ok) {
    const msg = data?.detail ? String(data.detail) : "Request failed";
    throw new Error(msg);
  }
  return { payload, data };
}

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  predictBtn.disabled = true;
  setResult("Requesting prediction...", "muted");
  jsonDumpEl.textContent = "";

  try {
    const customer = formToCustomer(form);
    const threshold = Number(thresholdEl.value);
    const { payload, data } = await predict(customer, threshold);

    const prob = Number(data.churn_probability);
    const pred = Number(data.churn_prediction);

    const kind = pred === 1 ? "warn" : "ok";
    setResult(
      `Churn probability: ${prob.toFixed(4)} | Prediction: ${pred} (threshold ${data.threshold})`,
      kind
    );

    jsonDumpEl.textContent = pretty({ request: payload, response: data });
  } catch (err) {
    setResult(`Error: ${err.message}`, "warn");
  } finally {
    predictBtn.disabled = false;
  }
});

fillExample2Btn.addEventListener("click", () => {
  // Higher-risk-ish example (not guaranteed; depends on model)
  const set = (name, value) => {
    const el = form.querySelector(`[name="${name}"]`);
    if (el) el.value = value;
  };

  set("CreditScore", 450);
  set("Geography", "Germany");
  set("Gender", "Female");
  set("Age", 50);
  set("Tenure", 1);
  set("Balance", 120000);
  set("NumOfProducts", 1);
  set("HasCrCard", 0);
  set("IsActiveMember", 0);
  set("EstimatedSalary", 60000);
});

