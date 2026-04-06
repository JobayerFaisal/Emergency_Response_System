import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    host: '0.0.0.0',
    port: 3000,
    proxy: {
      // Proxy REST API calls to dashboard_api service
      '/api': {
        target:    'http://dashboard_api:8005',
        changeOrigin: true,
      },
      // Proxy WebSocket connection to dashboard_api service
      '/ws': {
        target:    'ws://dashboard_api:8005',
        ws:        true,
        changeOrigin: true,
      },
    },
  },
  build: {
    outDir:    'dist',
    sourcemap: false,
  },
})
