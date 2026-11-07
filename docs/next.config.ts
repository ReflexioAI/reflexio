import path from "node:path";
import type { NextConfig } from "next";

const nextConfig: NextConfig = {
	experimental: { cpus: 2 },
	turbopack: {
		root: path.resolve(__dirname),
	},
};

export default nextConfig;
