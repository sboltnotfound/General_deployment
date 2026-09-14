const form = document.getElementById("predictionForm");
const fieldsContainer = document.getElementById("fields");
const predictButton = document.getElementById("predictButton");
const buttonText = document.getElementById("buttonText");
const spinner = document.getElementById("spinner");
const errorBox = document.getElementById("error");
const resultBox = document.getElementById("result");
const predictionValue = document.getElementById("predictionValue");
const targetLabel = document.getElementById("targetLabel");
const probabilitiesBox = document.getElementById("probabilities");
const modelInfo = document.getElementById("modelInfo");

let config = null;

function prettyName(name) {
    return name
        .replace(/_/g, " ")
        .replace(/([a-z])([A-Z])/g, "$1 $2")
        .replace(/\b\w/g, c => c.toUpperCase());
}

function createField(name, metadata = {}) {
    const wrapper = document.createElement("div");
    wrapper.className = "field";

    const label = document.createElement("label");
    label.htmlFor = `field-${name}`;
    label.textContent = prettyName(name);

    if (metadata.required !== false) {
        const required = document.createElement("span");
        required.className = "required";
        required.textContent = " *";
        label.appendChild(required);
    }

    const type = metadata.type || "string";
    let input;

    if (Array.isArray(metadata.options)) {
        input = document.createElement("select");

        const placeholder = document.createElement("option");
        placeholder.value = "";
        placeholder.textContent = "Select...";
        placeholder.disabled = metadata.required !== false;
        placeholder.selected = true;
        input.appendChild(placeholder);

        for (const option of metadata.options) {
            const optionElement = document.createElement("option");
            optionElement.value = option;
            optionElement.textContent = option;
            input.appendChild(optionElement);
        }
    } else {
        input = document.createElement("input");

        if (type === "numeric") {
            input.type = "number";
            input.step = "any";
        } else if (type === "integer") {
            input.type = "number";
            input.step = "1";
        } else if (type === "date") {
            input.type = "date";
        } else if (type === "boolean") {
            input.type = "checkbox";
        } else {
            input.type = "text";
        }
    }

    input.id = `field-${name}`;
    input.name = name;

    if (metadata.required !== false && input.type !== "checkbox") {
        input.required = true;
    }

    if (metadata.placeholder && input.type !== "checkbox") {
        input.placeholder = metadata.placeholder;
    }

    if (metadata.min !== undefined) input.min = metadata.min;
    if (metadata.max !== undefined) input.max = metadata.max;

    wrapper.appendChild(label);
    wrapper.appendChild(input);

    return wrapper;
}

async function loadConfig() {
    try {
        const response = await fetch("/api/config");

        if (!response.ok) {
            throw new Error("Could not load model configuration.");
        }

        config = await response.json();

        fieldsContainer.innerHTML = "";

        for (const feature of config.feature_columns) {
            const metadata = config.feature_metadata?.[feature] || {};
            fieldsContainer.appendChild(createField(feature, metadata));
        }

        const badges = [
            config.problem_type,
            config.training_rows ? `${config.training_rows} training rows` : null,
            config.sklearn_version ? `sklearn ${config.sklearn_version}` : null
        ].filter(Boolean);

        modelInfo.innerHTML = badges
            .map(x => `<span class="badge">${escapeHtml(String(x))}</span>`)
            .join("");

        if (badges.length) {
            modelInfo.classList.remove("hidden");
        }

    } catch (error) {
        showError(error.message);
    }
}

function collectFormData() {
    const data = {};

    for (const feature of config.feature_columns) {
        const metadata = config.feature_metadata?.[feature] || {};
        const input = document.getElementById(`field-${feature}`);

        if (metadata.type === "boolean") {
            data[feature] = input.checked;
        } else {
            data[feature] = input.value;
        }
    }

    return data;
}

async function predict(event) {
    event.preventDefault();
    clearMessages();

    const payload = collectFormData();

    predictButton.disabled = true;
    buttonText.textContent = "Predicting";
    spinner.classList.remove("hidden");

    try {
        const response = await fetch("/api/predict", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify(payload)
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.detail || "Prediction failed.");
        }

        predictionValue.textContent = formatValue(data.prediction);
        targetLabel.textContent = data.target
            ? `Target: ${data.target}`
            : "";

        if (
            data.probabilities &&
            data.classes &&
            data.probabilities.length === data.classes.length
        ) {
            probabilitiesBox.innerHTML = "<strong>Class probabilities</strong>";

            data.classes.forEach((className, index) => {
                const row = document.createElement("div");
                row.className = "probability-row";

                const label = document.createElement("span");
                label.textContent = String(className);

                const probability = document.createElement("span");
                probability.textContent =
                    `${(data.probabilities[index] * 100).toFixed(2)}%`;

                row.appendChild(label);
                row.appendChild(probability);
                probabilitiesBox.appendChild(row);
            });

            probabilitiesBox.classList.remove("hidden");
        }

        resultBox.classList.remove("hidden");

    } catch (error) {
        showError(error.message);
    } finally {
        predictButton.disabled = false;
        buttonText.textContent = "Predict";
        spinner.classList.add("hidden");
    }
}

function formatValue(value) {
    if (typeof value === "number") {
        return Number.isInteger(value)
            ? String(value)
            : value.toFixed(4).replace(/0+$/, "").replace(/\.$/, "");
    }

    if (typeof value === "object") {
        return JSON.stringify(value);
    }

    return String(value);
}

function escapeHtml(value) {
    return value
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;");
}

function showError(message) {
    errorBox.textContent = message;
    errorBox.classList.remove("hidden");
}

function clearMessages() {
    errorBox.classList.add("hidden");
    resultBox.classList.add("hidden");
    probabilitiesBox.classList.add("hidden");
}

form.addEventListener("submit", predict);
loadConfig();
