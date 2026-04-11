import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";
import { Providers } from "./providers";
import { Sidebar } from "@/components/layout/sidebar";
import { TopBar } from "@/components/layout/top-bar";

const inter = Inter({
  variable: "--font-sans",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "Reflexio API Docs",
  description: "Interactive API documentation for Reflexio",
  icons: {
    icon: "/reflexio_fav.svg",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className={`${inter.variable} h-full`} suppressHydrationWarning>
      <body className="h-full flex flex-col antialiased font-sans">
        <Providers>
          <TopBar />
          <div className="flex flex-1 min-h-0">
            <aside className="hidden lg:block w-64 border-r border-border shrink-0">
              <Sidebar />
            </aside>
            <main className="flex-1 min-w-0 flex flex-col">{children}</main>
          </div>
        </Providers>
      </body>
    </html>
  );
}
