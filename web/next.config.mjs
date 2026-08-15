/** @type {import('next').NextConfig} */
const nextConfig = {
  // The engine and level data live outside web/, shared with the Python trainer.
  outputFileTracingRoot: new URL('..', import.meta.url).pathname,
};

export default nextConfig;
