const { createProxyMiddleware } = require('http-proxy-middleware');

module.exports = (app) => {
  const proxy = createProxyMiddleware({
    target: 'http://127.0.0.1:8000',
    changeOrigin: true,
  });

  // Filter before proxying so both v2 and v3 preserve frontend routes.
  app.use((req, res, next) => {
    const path = req.path;
    const isBackendPath = ['/api', '/static/uploads', '/static/explain'].some(
      (prefix) => path === prefix || path.startsWith(`${prefix}/`)
    );
    return isBackendPath ? proxy(req, res, next) : next();
  });
};