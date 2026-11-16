import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// base: the repo name, because this deploys to GitHub Pages at
// /<repo>/. Override with BASE_PATH when serving from a domain root.
export default defineConfig({
  plugins: [react()],
  base: process.env.BASE_PATH ?? '/tn-peak-demand/',
  build: { outDir: 'dist', assetsInlineLimit: 4096 },
})
