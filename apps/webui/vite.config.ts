import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      // Forward /api/v1 to the FastAPI backend when VITE_ENABLE_MOCK=false
      // Set VITE_ENABLE_MOCK=false in .env.local to disable MSW and use real API
      '/api/v1': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
    },
  },
})
