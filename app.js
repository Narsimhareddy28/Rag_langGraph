// --- ELEMENTS ---
const uploadForm = document.getElementById("upload-form");
const fileInput = document.getElementById("file-input");
const chatForm = document.getElementById("chat-form");
const userInput = document.getElementById("user-input");
const chatWindow = document.getElementById("chat-window");
const uploadStatus = document.getElementById("upload-status");

// --- CHAT HISTORY ---
let chatHistory = [];

// --- FILE UPLOAD HANDLER ---
uploadForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const file = fileInput.files[0];
  if (!file) {
    alert("❗ Please select a PDF file first.");
    return;
  }

  const formData = new FormData();
  formData.append("file", file);

  try {
    console.log("⏳ Uploading:", file.name);

    const res = await fetch("http://localhost:8000/upload", {
      method: "POST",
      body: formData,
    });

    if (!res.ok) {
      const errorData = await res.json();
      alert("❌ Upload failed: " + errorData.message);
      return;
    }

    const data = await res.json();
    console.log("✅ Upload response:", data);
    uploadStatus.textContent = data.message;
    uploadStatus.style.color = "green";
    alert(data.message); // optional: visible confirmation

  } catch (err) {
    console.error("❌ Upload error:", err);
    alert("❌ Upload failed: " + err.message);
    uploadStatus.textContent = "❌ Upload failed.";
    uploadStatus.style.color = "red";
  }
});

// --- CHAT SUBMISSION HANDLER ---
chatForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const input = userInput.value.trim();
  if (!input) return;

  appendMessage("user", input);
  userInput.value = "";

  try {
    const res = await fetch("http://localhost:8000/chat", {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({ 
        message: input,
        history: chatHistory
      })
    });

    if (!res.ok) {
      throw new Error(`HTTP error! status: ${res.status}`);
    }

    const data = await res.json();
    appendMessage("bot", data.response);
    
    // Update chat history
    chatHistory = data.messages;

  } catch (err) {
    console.error("❌ Chat error:", err);
    appendMessage("bot", "❌ Failed to get a response. Check your server.");
  }
});

// --- CHAT UI HELPERS ---
function appendMessage(sender, text) {
  const div = document.createElement("div");
  div.classList.add("message", sender);
  div.innerHTML = `<strong>${sender === "user" ? "You" : "Bot"}:</strong> ${text}`;
  chatWindow.appendChild(div);
  chatWindow.scrollTop = chatWindow.scrollHeight;
}