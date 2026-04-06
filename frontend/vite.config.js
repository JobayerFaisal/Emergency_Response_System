import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000,
    proxy: {
      '/api': 'http://localhost:8005',
      '/ws': {
        target: 'ws://localhost:8005',
        ws: true,
      }
    }
  }
})