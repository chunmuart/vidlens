import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

// Vite 配置参考：https://vite.dev/config/
// 端口固定为 5174，与后端 CORS_ALLOWED_ORIGINS 和 start-vidlens.ps1 保持一致；
// strictPort 让端口被占用时直接报错，而不是悄悄换到 5173 导致跨域失败。
export default defineConfig({
  plugins: [vue()],
  server: {
    host: '127.0.0.1',
    port: 5174,
    strictPort: true,
  },
})
