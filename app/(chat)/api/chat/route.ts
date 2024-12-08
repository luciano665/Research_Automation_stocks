import {
  type Message,
  StreamData,
  //convertToCoreMessages,
  //streamObject,
  streamText,
} from "ai";
//import { z } from "zod";
import { groq } from '@ai-sdk/groq';

import { auth } from "@/app/(auth)/auth";
import { customModel } from "@/lib/ai";
//import { models } from "@/lib/ai/models";
//import { systemPrompt } from "@/lib/ai/prompts";
import {
  deleteChatById,
  getChatById,
  saveChat,
  //getDocumentById,
  //saveChat,
  //saveDocument,
  saveMessages,
  //saveSuggestions,
} from "@/lib/db/queries";
//import type { Suggestion } from "@/lib/db/schema";
import {
  generateUUID,
  getMostRecentUserMessage,
  sanitizeResponseMessages,
} from "@/lib/utils";

//import { generateTitleFromUserMessage } from "../../actions";
//import { getHuggingFaceEmbeddings } from "@/lib/ai/embeddings";
import { getPineconeClient } from "@/lib/ai/pinecone";
import { generateTitleFromUserMessage } from "../../actions";

export const maxDuration = 60;

const namespaces = [
  "stock-descriptions",
];

type AllowedTools =
  | "createDocument"
  | "updateDocument"
  | "requestSuggestions"
  | "getWeather";

const blocksTools: AllowedTools[] = [
  "createDocument",
  "updateDocument",
  "requestSuggestions",
];

const weatherTools: AllowedTools[] = ["getWeather"];

const allTools: AllowedTools[] = [...blocksTools, ...weatherTools];

const MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2";
const HF_API_URL = `https://api-inference.huggingface.co/pipeline/feature-extraction/${MODEL_ID}`;

async function getEmbeddings(text: string) {
  console.log("📤 Sending text to HF:", text.substring(0, 100) + "...");

  const payload = {
    inputs: text,
    options: { wait_for_model: true },
  };
  console.log("📦 Request payload:", payload);

  const response = await fetch(HF_API_URL, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${process.env.HUGGINGFACE_API_KEY}`,
      "Content-Type": "application/json", // Added this header explicitly
    },
    body: JSON.stringify(payload),
  });

  if (!response.ok) {
    const errorText = await response.text();
    console.error("🚫 HF API Error:", {
      status: response.status,
      statusText: response.statusText,
      error: errorText,
    });
    throw new Error(
      `Failed to get embeddings: ${response.statusText} - ${errorText}`
    );
  }

  const result = await response.json();
  console.log(
    "📥 Raw HF response:",
    JSON.stringify(result).substring(0, 200) + "..."
  );

  return result;
}

export async function POST(request: Request) {
  try {
    console.log("🟦 Starting POST request");

    const { messages, id }: { messages: Array<Message>; id: string } =
      await request.json();
    console.log("📨 Received request data:", {
      messageCount: messages.length,
      chatId: id,
    });

    const session = await auth();
    console.log("🔐 Auth session:", {
      authenticated: !!session,
      userId: session?.user?.id,
    });

    if (!session || !session.user || !session.user.id) {
      console.log("❌ Authentication failed");
      return new Response("Unauthorized", { status: 401 });
    }

    const lastMessage = getMostRecentUserMessage(messages);
    console.log("💭 Last user message:", lastMessage?.content);

    if (!lastMessage) {
      console.log("❌ No user message found");
      return new Response("No user message found", { status: 400 });
    }
    console.log("🔄 Getting chat from database");
    const chat = await getChatById({ id });

    if (!chat) {
      console.log("🔄 Generating title from user message");
      const title = await generateTitleFromUserMessage({
        message: lastMessage,
      });
      console.log("🔄 Saving chat to database");
      await saveChat({ id, userId: session.user.id, title });
    }

    console.log("🔄 Saving user message to database");
    await saveMessages({
      messages: [
        {
          ...lastMessage,
          id: generateUUID(),
          createdAt: new Date(),
          chatId: id,
        },
      ],
    });

    console.log("🔄 Getting embeddings from Hugging Face");
    const queryEmbedding = await getEmbeddings(lastMessage.content);
    console.log("✅ Embeddings generated\n", queryEmbedding);

    console.log("🔄 Connecting to Pinecone");
    const pinecone = await getPineconeClient();
    console.log("✅ Pinecone client ready");

    console.log("🔄 Querying all namespaces");
    const namespaceResults = await Promise.all(
      namespaces.map(async (namespace) => {
        console.log(`  📍 Querying namespace: ${namespace}`);
        const testIndex = pinecone.Index("codebase-rag").namespace(namespace);
        const queryResponse = await testIndex.query({
          vector: queryEmbedding,
          topK: 5,
          includeMetadata: true,
        });

        const averageScore =
          queryResponse.matches.reduce(
            (acc, match) => acc + (match.score || 0),
            0
          ) / queryResponse.matches.length;

        console.log(`  📊 Average score for ${namespace}: ${averageScore}`);
        return {
          namespace,
          score: averageScore,
          matches: queryResponse.matches,
        };
      })
    );

    const bestNamespaceResult = namespaceResults.reduce((best, current) =>
      current.score > best.score ? current : best
    );
    console.log("🏆 Best matching namespace:", {
      namespace: bestNamespaceResult.namespace,
      score: bestNamespaceResult.score,
    });

    const contexts = bestNamespaceResult.matches.map(
      (match) => match.metadata?.text || ""
    );
    console.log("📚 Retrieved contexts:", {
      count: contexts.length,
    });

    const augmentedQuery = `
      <CONTEXT>
      ${contexts.join("\n\n-------\n\n")}
      </CONTEXT>

      MY QUESTION:
      ${lastMessage.content}`;
    console.log("📝 Created augmented query");

    const systemPrompt = `
You are a financial expert assistant designed to handle queries about financial data, stocks, and investments while maintaining a clear, structured response in various formats. Your primary objective is to present data accurately and concisely in the requested format.

General Rules for Responses:
1. Identify the User's Formatting Need:
   - Detect if the user asks for a specific format (e.g., tables, JSON, step-by-step explanations).
   - If no format is specified, choose the most appropriate format based on the type of data.

2. Supported Formats:
   - **Markdown Tables**: Use for comparisons or tabular data.
   - **JSON**: Use for structured or programmatically consumable data.
   - **Plain Text (Steps)**: Use for explanations or instructions.
   - **Code Blocks**: Use when the response includes examples of scripts, queries, or formulas.

3. Formatting Rules:
   - Use proper Markdown for tables.
   - Indent JSON properly for readability.
   - Clearly number steps in step-by-step explanations.
   - For code blocks, wrap code in triple backticks (\`\`\`) and specify the language (e.g., \`\`\`javascript).

4. Respond as a Financial Expert:
   - Include relevant financial metrics (e.g., market cap, P/E ratio, dividend yield) in responses.
   - Provide actionable insights wherever possible.
   - Ensure accuracy by performing computations if necessary.

5. Example Responses for Each Format:

   - Markdown Table:
     \`\`\`
     | Ticker | Company Name      | Sector       | Market Cap | P/E Ratio | Dividend Yield |
     |--------|-------------------|--------------|------------|-----------|----------------|
     | AAPL   | Apple Inc.        | Technology   | $2.5T      | 25        | 0.6%           |
     | MSFT   | Microsoft Corp.   | Technology   | $2.2T      | 30        | 0.8%           |
     \`\`\`

   - JSON:
     \`\`\`json
     {
       "stocks": [
         {
           "ticker": "AAPL",
           "company": "Apple Inc.",
           "sector": "Technology",
           "marketCap": "$2.5T",
           "peRatio": 25,
           "dividendYield": "0.6%"
         },
         {
           "ticker": "MSFT",
           "company": "Microsoft Corp.",
           "sector": "Technology",
           "marketCap": "$2.2T",
           "peRatio": 30,
           "dividendYield": "0.8%"
         }
       ]
     }
     \`\`\`

   - Step-by-Step Explanation:
     \`\`\`
     1. Analyze the Technology sector for companies with a market cap greater than $1T.
     2. Identify those with P/E ratios below 30 to find potentially undervalued stocks.
     3. Filter stocks offering a dividend yield greater than 0.5%.
     4. Review the final shortlist: AAPL, MSFT, etc.
     \`\`\`

   - Code Block:
     \`\`\`javascript
     const filterStocks = (stocks) => {
       return stocks.filter(stock => stock.marketCap > 1e12 && stock.peRatio < 30 && stock.dividendYield > 0.5);
     };

     const stocks = [
       { ticker: "AAPL", marketCap: 2.5e12, peRatio: 25, dividendYield: 0.6 },
       { ticker: "MSFT", marketCap: 2.2e12, peRatio: 30, dividendYield: 0.8 }
     ];

     console.log(filterStocks(stocks));
     \`\`\`

6. Clarify Ambiguities:
   - If the user’s request is vague, ask for clarification before generating the response.

7. Example Queries You Should Handle:
   - "Show me a table of the top 5 NYSE companies by market cap."
   - "Generate JSON data for tech companies with P/E < 20."
   - "Explain step-by-step how to calculate a stock's intrinsic value."
   - "Provide a Python code snippet to calculate compound annual growth rate (CAGR)."

IMPORTANT:
- Always ensure responses are professional, accurate, and tailored to the user's intent.
- Do not mention internal processes or the database source in your responses.
`;
    const llmMessages = [
      { role: "system" as const, content: systemPrompt },
      ...messages.slice(0, -1).map((msg) => ({
        role: msg.role as "user" | "assistant",
        content: msg.content,
      })),
      { role: "user" as const, content: augmentedQuery },
    ];

    const streamingData = new StreamData();

    console.log("🤖 Calling LLM with streamText");
    const result = await streamText({
      model: groq('llama-3.3-70b-versatile'),
      system: systemPrompt,
      messages: llmMessages,
      maxSteps: 5,
      onFinish: async ({ responseMessages }) => {
        console.log("✨ Stream finished, processing response messages");
        if (session.user?.id) {
          try {
            console.log("🔄 Sanitizing response messages");
            const responseMessagesWithoutIncompleteToolCalls =
              sanitizeResponseMessages(responseMessages);

            console.log("💾 Saving messages to database");
            await saveMessages({
              messages: responseMessagesWithoutIncompleteToolCalls.map(
                (message) => {
                  const messageId = generateUUID();
                  console.log(`  📝 Processing message: ${message.role}`);

                  if (message.role === "assistant") {
                    console.log(
                      `  🏷️ Appending message annotation: ${messageId}`
                    );
                    streamingData.appendMessageAnnotation({
                      messageIdFromServer: messageId,
                    });
                  }

                  return {
                    id: messageId,
                    chatId: id,
                    role: message.role,
                    content: message.content,
                    createdAt: new Date(),
                  };
                }
              ),
            });
            console.log("✅ Messages saved successfully");
          } catch (error) {
            console.error("❌ Failed to save chat:", error);
          }
        }

        console.log("👋 Closing stream");
        streamingData.close();
      },
    });

    console.log("🔄 Converting to data stream response");
    const response = result.toDataStreamResponse({
      data: streamingData,
    });
    console.log("✅ Response ready to send");

    return response;
  } catch (error) {
    console.error("❌ Error in chat route:", error);
    return new Response(JSON.stringify({ error: "Internal Server Error" }), {
      status: 500,
      headers: { "Content-Type": "application/json" },
    });
  }
}

export async function DELETE(request: Request) {
  const { searchParams } = new URL(request.url);
  const id = searchParams.get("id");

  if (!id) {
    return new Response("Not Found", { status: 404 });
  }

  const session = await auth();

  if (!session || !session.user) {
    return new Response("Unauthorized", { status: 401 });
  }

  try {
    const chat = await getChatById({ id });

    if (chat.userId !== session.user.id) {
      return new Response("Unauthorized", { status: 401 });
    }

    await deleteChatById({ id });

    return new Response("Chat deleted", { status: 200 });
  } catch (error) {
    return new Response("An error occurred while processing your request", {
      status: 500,
    });
  }
}
