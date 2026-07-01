import type { Metadata } from "next";
import { Geist } from "next/font/google";
import "./globals.css";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "Victoria — Private Chat",
  description: "AI Girlfriend Chat Demo",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className={`${geistSans.variable} dark`}>
      <body className="h-screen overflow-hidden bg-[#0d0d0d] text-[#e5e5e5] antialiased">
        {children}
      </body>
    </html>
  );
}
