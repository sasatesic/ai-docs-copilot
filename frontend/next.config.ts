import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  /* config options here */
  
  // Keeps your experimental compiler setting
  reactCompiler: true,

  // Optimized for Docker deployment
  output: "standalone",

  // The Proxy Configuration
  async rewrites() {
    return [
      {
        source: "/api/:path*",
        destination: "http://127.0.0.1:8000/:path*", // Forward requests to Python Backend
      },
    ];
  },
};

export default nextConfig;