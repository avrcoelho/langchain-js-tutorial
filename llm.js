import { ChatGoogle } from "@langchain/google";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import dotenv from "dotenv";

dotenv.config({ path: ".env", override: true });

// Create model
const model = new ChatGoogle({
  model: "gemini-pro-latest",
  temperature: 0.7,
  verbose: false,
});

model.invoke("Hello World").then((response) => {
  console.log(response.content);
});
