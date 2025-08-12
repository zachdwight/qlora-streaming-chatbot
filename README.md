# qlora-streaming-chatbot

## Quick Start
Follow these simple steps to get your QLoRA streaming chatbot up and running:

1. **Add your adapter**  
   Place your fine-tuned adapter folder (e.g., `tinyllama-qlora-finetuned`) inside the `backend/` directory.

2. **Prepare the frontend**  
   Navigate to the `frontend/` folder and run:  
   ```bash
   npm install
   npm run build
     ```

3. **Build and launch with Docker Compose**  
   From the root of the repository, run: 
   ```bash
   docker-compose up --build
     ```

4. **Start Chatting!**  
   Open your browser and go to http://localhost:8080 to try out the chatbot interface.

