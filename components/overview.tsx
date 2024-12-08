import { motion } from "framer-motion";
import Link from "next/link";

import { MessageIcon, VercelIcon } from "./icons";

export const Overview = () => {
  return (
    <motion.div
      key="overview"
      className="max-w-3xl mx-auto md:mt-20"
      initial={{ opacity: 0, scale: 0.98 }}
      animate={{ opacity: 1, scale: 1 }}
      exit={{ opacity: 0, scale: 0.98 }}
      transition={{ delay: 0.5 }}
    >
      <div className="rounded-xl p-6 flex flex-col gap-8 leading-relaxed text-center max-w-xl">
        <p className="flex flex-row justify-center gap-4 items-center">
          <MessageIcon size={32} />
        </p>
        <p>
          This is an chatbot model tuned for request based on research related
          to the Financial market. We have stored more than 8000 vectors in our
          DB from real and up-to-date information. Please restarin yourself to
          use this model only for financial inquires more likely stocks
          information. We use a template that is{" "}
          <Link
            className="font-medium underline underline-offset-4"
            href="https://github.com/vercel/ai-chatbot"
            target="_blank"
          >
            Open Source
          </Link>
        </p>
      </div>
    </motion.div>
  );
};
