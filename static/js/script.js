// 🔍 Function to Detect Deepfake
function detectDeepfake() {
    let imageInput = document.getElementById("imageInput").files[0];

    if (!imageInput) {
        alert("Please select an image to upload!");
        return;
    }

    let formData = new FormData();
    formData.append("file", imageInput);

    document.getElementById("deepfakeResult").innerText = "🔄 Processing...";

    fetch("/deepfake/detect", {
        method: "POST",
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        setTimeout(() => { // Adds a delay to simulate loading
            document.getElementById("deepfakeResult").innerText = "🧠 Result: " + data.result;
        }, 1000);
    })
    .catch(error => console.error("Error:", error));
}

// 📰 Function to Verify Fake News
function verifyNews() {
    let newsText = document.getElementById("newsInput").value;

    if (!newsText.trim()) {
        alert("Please enter some text to verify.");
        return;
    }

    document.getElementById("newsResult").innerText = "🔄 Checking authenticity...";

    fetch("/fakenews/verify", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ article_text: newsText })
    })
    .then(response => response.json())
    .then(data => {
        setTimeout(() => {
            document.getElementById("newsResult").innerText = "🔍 Authenticity Score: " + data.authenticity + "%";
        }, 1000);
    })
    .catch(error => console.error("Error:", error));
}
