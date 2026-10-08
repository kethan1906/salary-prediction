"use strict";
const $ = (id) => document.getElementById(id);

function notice(message, isError) {
  const el = $("status");
  el.textContent = message || "";
  el.className = "notice" + (isError ? " error" : "");
  el.hidden = !message;
}

function fillSelect(select, values) {
  select.replaceChildren();
  values.forEach((v) => {
    const option = document.createElement("option");
    option.value = v;
    option.textContent = v;
    select.appendChild(option);
  });
}

async function loadOptions() {
  try {
    const response = await fetch("/api/options");
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "Could not load options");
    fillSelect($("gender"), data.genders);
    fillSelect($("education"), data.education_levels);
    const list = $("titles");
    list.replaceChildren();
    data.job_titles.forEach((t) => {
      const option = document.createElement("option");
      option.value = t;
      list.appendChild(option);
    });
    const r = data.training_ranges;
    $("age").placeholder = r.Age[0] + "-" + r.Age[1] + " in training data";
    $("years").placeholder = r.Years_of_Experience[0] + "-" + r.Years_of_Experience[1] + " in training data";
  } catch (error) {
    notice(error.message, true);
    $("form").querySelector("button").disabled = true;
  }
}

async function predict(event) {
  event.preventDefault();
  notice("");
  const payload = {
    Age: $("age").value,
    Gender: $("gender").value,
    Education_Level: $("education").value,
    Job_Title: $("title").value,
    Years_of_Experience: $("years").value,
  };
  try {
    const response = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "Prediction failed");
    $("amount").textContent = Math.round(data.predicted_salary).toLocaleString();
    $("error-range").textContent =
      "Typical error on held-out data: about " + Math.round(data.typical_error_mae).toLocaleString() +
      " (mean absolute error).";
    const ul = $("warnings");
    ul.replaceChildren();
    data.warnings.forEach((w) => {
      const li = document.createElement("li");
      li.textContent = w;
      ul.appendChild(li);
    });
    $("result").hidden = false;
  } catch (error) {
    $("result").hidden = true;
    notice(error.message, true);
  }
}

document.addEventListener("DOMContentLoaded", () => {
  loadOptions();
  $("form").addEventListener("submit", predict);
});
