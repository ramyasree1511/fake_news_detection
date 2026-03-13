const newsUrlInput = document.getElementById("newsUrl");
const newsTextInput = document.getElementById("newsText");
const checkNewsBtn = document.getElementById("checkNewsBtn");
const loading = document.getElementById("loading");
const resultCard = document.getElementById("resultCard");
const predictionText = document.getElementById("predictionText");
const confidenceText = document.getElementById("confidenceText");
const articleTitle = document.getElementById("articleTitle");
const wordCount = document.getElementById("wordCount");
const previewText = document.getElementById("previewText");
const reasonText = document.getElementById("reasonText");
const errorText = document.getElementById("errorText");
const sourceText = document.getElementById("sourceText");
const modelAccuracyResult = document.getElementById("modelAccuracyResult");
const urlModeBtn = document.getElementById("urlModeBtn");
const textModeBtn = document.getElementById("textModeBtn");
const textInputGroup = document.getElementById("textInputGroup");

let activeMode = "url";

function setMode(mode) {
    activeMode = mode;
    const isUrlMode = mode === "url";

    urlModeBtn.classList.toggle("active", isUrlMode);
    textModeBtn.classList.toggle("active", !isUrlMode);
    newsUrlInput.parentElement.classList.toggle("hidden", !isUrlMode);
    textInputGroup.classList.toggle("hidden", isUrlMode);
    errorText.classList.add("hidden");
}

function toggleLoading(isLoading) {
    loading.classList.toggle("hidden", !isLoading);
    checkNewsBtn.disabled = isLoading;
    checkNewsBtn.textContent = isLoading ? "Checking..." : "Check News";
}

function showError(message) {
    const scraperBlocked = message.toLowerCase().includes("blocking automated scraping");
    errorText.textContent = scraperBlocked
        ? `${message} Switch to "Paste Article Text" for this article.`
        : message;
    errorText.classList.remove("hidden");
    resultCard.classList.add("hidden");
}

function showResult(data) {
    const label = data.label.toUpperCase();
    const resultClass = label === "REAL" ? "real" : "fake";

    predictionText.textContent = label;
    predictionText.style.color = label === "REAL" ? "#1c9c63" : "#d64545";
    confidenceText.textContent = `${data.confidence}%`;
    articleTitle.textContent = data.title || "Untitled Article";
    wordCount.textContent = data.word_count;
    previewText.textContent = data.preview;
    reasonText.textContent = data.reason || "Model prediction only";
    sourceText.textContent = data.source === "url" ? "Analyzed from URL" : "Analyzed from pasted text";
    modelAccuracyResult.textContent = `${data.model_accuracy}%`;

    resultCard.classList.remove("hidden", "real", "fake");
    resultCard.classList.add(resultClass);
    errorText.classList.add("hidden");
}

async function checkNews() {
    const url = newsUrlInput.value.trim();
    const text = newsTextInput.value.trim();

    if (activeMode === "url" && !url) {
        showError("Please enter a valid news article URL.");
        return;
    }

    if (activeMode === "text" && !text) {
        showError("Please paste article text to analyze.");
        return;
    }

    toggleLoading(true);
    errorText.classList.add("hidden");

    try {
        const response = await fetch("/predict", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify(activeMode === "url" ? { url } : { text }),
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || "Unable to analyze this article.");
        }

        showResult(data);
    } catch (error) {
        showError(error.message);
    } finally {
        toggleLoading(false);
    }
}

checkNewsBtn.addEventListener("click", checkNews);
urlModeBtn.addEventListener("click", () => setMode("url"));
textModeBtn.addEventListener("click", () => setMode("text"));

newsUrlInput.addEventListener("keydown", (event) => {
    if (event.key === "Enter") {
        checkNews();
    }
});
