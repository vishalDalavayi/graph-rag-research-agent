import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Graph RAG Research Agent",
  description: "Upload a PDF and explore extracted knowledge graphs",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
