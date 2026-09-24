/**
 * FloraScan AI - Frontend Interactions & Dynamic Image Uploader
 */

document.addEventListener("DOMContentLoaded", () => {
  const dropzone = document.getElementById("dropzone");
  const fileInput = document.getElementById("fileInput");
  const dropzonePrompt = document.getElementById("dropzonePrompt");
  const previewContainer = document.getElementById("previewContainer");
  const imagePreview = document.getElementById("imagePreview");
  const fileName = document.getElementById("fileName");
  const fileSize = document.getElementById("fileSize");
  const removeFileBtn = document.getElementById("removeFileBtn");
  const submitBtn = document.getElementById("submitBtn");
  const uploadForm = document.getElementById("uploadForm");
  const loadingOverlay = document.getElementById("loadingOverlay");
  const loadingText = document.getElementById("loadingText");
  const loadingSubtext = document.getElementById("loadingSubtext");

  if (!dropzone || !fileInput) return;

  const validExtensions = ["jpg", "jpeg", "png", "webp"];
  const maxBytes = 16 * 1024 * 1024; // 16MB

  // Drag and drop event listeners
  ["dragenter", "dragover"].forEach((eventName) => {
    dropzone.addEventListener(eventName, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropzone.classList.add("dragover");
    });
  });

  ["dragleave", "drop"].forEach((eventName) => {
    dropzone.addEventListener(eventName, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropzone.classList.remove("dragover");
    });
  });

  // Handle dropped files
  dropzone.addEventListener("drop", (e) => {
    const files = e.dataTransfer.files;
    if (files.length > 0) {
      handleFile(files[0]);
    }
  });

  // Handle selected files
  fileInput.addEventListener("change", (e) => {
    if (fileInput.files.length > 0) {
      handleFile(fileInput.files[0]);
    }
  });

  function handleFile(file) {
    const ext = file.name.split(".").pop().toLowerCase();
    if (!validExtensions.includes(ext)) {
      alert("Unsupported file format! Please upload a JPG, JPEG, PNG, or WEBP leaf image.");
      resetUploader();
      return;
    }

    if (file.size > maxBytes) {
      alert("File size exceeds 16MB limit. Please choose a smaller image.");
      resetUploader();
      return;
    }

    // Set preview
    const reader = new FileReader();
    reader.onload = (e) => {
      imagePreview.src = e.target.result;
      fileName.textContent = file.name;
      fileSize.textContent = formatBytes(file.size);

      dropzonePrompt.classList.add("hidden");
      previewContainer.classList.remove("hidden");
      submitBtn.disabled = false;
    };
    reader.readAsDataURL(file);
  }

  function resetUploader() {
    fileInput.value = "";
    imagePreview.src = "";
    fileName.textContent = "";
    fileSize.textContent = "";
    previewContainer.classList.add("hidden");
    dropzonePrompt.classList.remove("hidden");
    submitBtn.disabled = true;
  }

  if (removeFileBtn) {
    removeFileBtn.addEventListener("click", (e) => {
      e.stopPropagation();
      resetUploader();
    });
  }

  // Handle form submission and animated progress
  if (uploadForm) {
    uploadForm.addEventListener("submit", (e) => {
      if (!fileInput.files || fileInput.files.length === 0) {
        e.preventDefault();
        alert("Please select a leaf image first.");
        return;
      }

      loadingOverlay.classList.remove("hidden");
      submitBtn.disabled = true;

      // Realistic multistage progress messages
      const statusSteps = [
        { title: "Uploading Specimen...", sub: "Validating leaf resolution and RGB color channels..." },
        { title: "Extracting Visual Features...", sub: "Passing specimen through deep convolutional feature extractors..." },
        { title: "Computing Explainability Heatmap...", sub: "Calculating Grad-CAM gradients to pinpoint lesions..." },
        { title: "Consulting Pathology Knowledge Base...", sub: "Formulating curative and preventive agricultural advisory..." }
      ];

      let stepIndex = 0;
      const interval = setInterval(() => {
        stepIndex++;
        if (stepIndex < statusSteps.length) {
          loadingText.textContent = statusSteps[stepIndex].title;
          loadingSubtext.textContent = statusSteps[stepIndex].sub;
        } else {
          clearInterval(interval);
        }
      }, 1400);
    });
  }

  function formatBytes(bytes) {
    if (bytes === 0) return "0 Bytes";
    const k = 1024;
    const sizes = ["Bytes", "KB", "MB", "GB"];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + " " + sizes[i];
  }
});
