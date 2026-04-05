import type { Metadata } from "next";
import { Sora, JetBrains_Mono } from "next/font/google";
import "./globals.css";

const sora = Sora({
  variable: "--font-sora",
  subsets: ["latin"],
});

const jetbrains = JetBrains_Mono({
  variable: "--font-jetbrains",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "Axes — Misinformation & Content Authenticity Detection",
  description:
    "Analyze images, videos, documents, and URLs across three forensic axes: Content Authenticity, Contextual Consistency, and Source Credibility.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className={`${sora.variable} ${jetbrains.variable}`}>
      <body className="min-h-screen antialiased">{children}</body>
    </html>
  );
}
