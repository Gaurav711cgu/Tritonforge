import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "TritonForge — High-Performance GPU Kernel Compilation Workstation",
  description: "Enterprise GPU kernel compilation and performance optimization workstation built on OpenAI Triton.",
  icons: {
    icon: "/logo-highres.png",
    shortcut: "/logo-highres.png",
    apple: "/logo-highres.png",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html
      lang="en"
      className={`${geistSans.variable} ${geistMono.variable} h-full antialiased`}
    >
      <head>
        <title>TritonForge — High-Performance GPU Kernel Compilation Workstation</title>
        <link rel="icon" href="/logo-highres.png" type="image/png" />
        <link rel="apple-touch-icon" href="/logo-highres.png" />
      </head>
      <body className="min-h-full flex flex-col">{children}</body>
    </html>
  );
}
