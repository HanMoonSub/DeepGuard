const { createProxyMiddleware } = require('http-proxy-middleware');

module.exports = (app) => {
  app.use(
    createProxyMiddleware({
      target: 'http://127.0.0.1:8000',
      changeOrigin: true,
      pathFilter: (path) =>
        path.startsWith('/api') ||
        path.startsWith('/static/uploads') ||
        path.startsWith('/static/explain'),
    })
  );
};