# Diffuion-Model-Resources
Important resources of diffusion models

Here I collected a lot of important resources to study diffusion models

**Denoising Diffusion Probabilistic Models**

https://arxiv.org/abs/2006.11239

import os
import uuid
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from PIL import Image
import torch
from transformers import CLIPTokenizer
import model_loader
import pipeline
import random


app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "outputs"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

MODEL_PATHS = {
    "default": "../data/v1-5-pruned-emaonly.ckpt",
    "inkpunk": "../data/inkpunk-diffusion-v1.ckpt",
}

def get_device(user_choice):
    if user_choice == "cuda" and torch.cuda.is_available():
        return "cuda"
    return "cpu"

@app.route("/start-generation", methods=["POST"])
def generate_image():
    try:
        prompt = request.form.get("prompt", "").strip()
        mode = request.form.get("mode", "txt2img").strip()
        model_choice = request.form.get("model", "default").strip()
        strength = float(request.form.get("strength", "1.0"))
        device_choice = request.form.get("device", "cpu")
        device = get_device(device_choice)

        if not prompt:
            return jsonify({"status": "error", "message": "Prompt cannot be empty"}), 400

        if model_choice not in MODEL_PATHS:
            model_choice = "default"  
        model_file = MODEL_PATHS[model_choice]

        models = model_loader.preload_models_from_standard_weights(model_file, device)

        input_image = None
        if mode == "img2img":
            image_file = request.files.get("image")
            if image_file:
                image_path = os.path.join(UPLOAD_FOLDER, f"{uuid.uuid4()}.jpg")
                image_file.save(image_path)
                input_image = Image.open(image_path)

        output_image = pipeline.generate(
            prompt=prompt,
            uncond_prompt="",
            input_image=input_image,
            strength=0.4,
            do_cfg=True,
            cfg_scale=8,
            sampler_name="ddpm",
            n_inference_steps=24,
            seed = random.randint(0, 2**32 - 1),
            models=models,
            device="cuda" if torch.cuda.is_available() else "cpu",
            idle_device="cpu",
            tokenizer=CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32"),
        )

        output_filename = f"{uuid.uuid4()}.jpg"
        output_path = os.path.join(OUTPUT_FOLDER, output_filename)
        Image.fromarray(output_image).save(output_path, format="JPEG")

        return jsonify({"status": "success", "image_url": f"http://127.0.0.1:5000/get-image/{output_filename}"})

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/get-image/<filename>")
def get_generated_image(filename):
    file_path = os.path.join(OUTPUT_FOLDER, filename)
    if os.path.exists(file_path):
        return send_file(file_path, mimetype="image/jpeg")
    return jsonify({"status": "error", "message": "Image not found"}), 404


if __name__ == "__main__":
    app.run(debug=False)

import React, { useState } from "react";
import axios from "axios";
import { motion } from "framer-motion";
import "./App.css";

function App() {
  const [prompt, setPrompt] = useState("");
  const [image, setImage] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [mode, setMode] = useState("txt2img");
  const [selectedImage, setSelectedImage] = useState(null);
  const [previewImage, setPreviewImage] = useState(null);
  const [strength, setStrength] = useState(1.0);
  const [model, setModel] = useState("default");

  const handleGenerateImage = async () => {
    setLoading(true);
    setError(null);
    setImage(null);
    try {
      const formData = new FormData();
      formData.append("prompt", prompt);
      formData.append("mode", mode);
      formData.append("model", model);
      if (mode === "img2img" && selectedImage) {
        formData.append("image", selectedImage);
        formData.append("strength", strength);
      }
      const response = await axios.post(
        "http://127.0.0.1:5000/start-generation",
        formData
      );
      if (response.data.status === "success") {
        setImage(response.data.image_url);
      } else {
        setError(response.data.message || "Image generation failed.");
      }
    } catch (error) {
      setError("An error occurred while generating the image.");
    } finally {
      setLoading(false);
    }
  };

  const handleImageUpload = (event) => {
    const file = event.target.files[0];
    if (file) {
      setSelectedImage(file);
      setPreviewImage(URL.createObjectURL(file));
    }
  };

  const handleSaveImage = () => {
    if (image) {
      const link = document.createElement("a");
      link.href = image;
      link.download = "generated_image.png";
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
    }
  };

  return (
    <div className="App">
      {/* Header */}
      <header className="app-header">
        <img src="/prism.jpg" alt="PRISM Logo" className="prism-logo" />
        <h1 className="header-title">AI Image Generator</h1>
      </header>

      <p className="sub-text">Create stunning visuals with AI</p>

      <motion.div
        className="container"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
      >
        <motion.textarea
          className="prompt-input"
          placeholder="Describe what you want to generate..."
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
        ></motion.textarea>

        {/* Model Selection */}
        <div className="selection-container">
          <label>Model:</label>
          <select value={model} onChange={(e) => setModel(e.target.value)}>
            <option value="default">Default Model</option>
            <option value="inkpunk">Inkpunk Model</option>
          </select>
          <p>Selected Model: {model}</p>
        </div>

        {/* Mode Selection */}
        <div className="selection-container">
          <label>Mode:</label>
          <select value={mode} onChange={(e) => setMode(e.target.value)}>
            <option value="txt2img">Text to Image</option>
            <option value="img2img">Image to Image</option>
          </select>
          <p>Selected Mode: {mode}</p>
        </div>

        {mode === "img2img" && (
          <div className="image-upload">
            <label htmlFor="upload" className="upload-btn">
              Upload Image
            </label>
            <input
              id="upload"
              type="file"
              accept="image/*"
              onChange={handleImageUpload}
              style={{ display: "none" }}
            />
          </div>
        )}

        {/* Center Uploaded Image */}
        {mode === "img2img" && selectedImage && (
          <div className="selected-image-container">
            <img
              src={previewImage}
              alt="Selected for img2img"
              className="selected-image"
            />
          </div>
        )}

        <button
          onClick={handleGenerateImage}
          disabled={loading}
          className="generate-btn"
        >
          {loading ? "Generating..." : "Generate Image"}
        </button>

        {error && <div className="error">{error}</div>}

        {image && (
          <div className="generated-container">
            <img src={image} alt="Generated" className="generated-image" />
            <button onClick={handleSaveImage} className="save-btn">
              Save Image
            </button>
          </div>
        )}
      </motion.div>
    </div>
  );
}

export default App;
