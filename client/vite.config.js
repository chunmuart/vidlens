import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

// Vite 配置参考：https://vite.dev/config/
// 开发端口与 Docker 网页入口统一为 5174；本机单独启动 API 时需允许此来源。
// strictPort 让端口被占用时直接报错，而不是悄悄换到 5173 导致跨域失败。
export default defineConfig({
  plugins: [vue()],
  server: {
    host: '127.0.0.1',
    port: 5174,
    strictPort: true,
  },
})
