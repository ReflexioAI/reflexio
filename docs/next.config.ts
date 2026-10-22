import type { NextConfig } from "next";
import path from "node:path";

const nextConfig: NextConfig = {
  experimental: { cpus: 2 },
  turbopack: {
    root: path.resolve(__dirname),
  },
};

export default nextConfig;
