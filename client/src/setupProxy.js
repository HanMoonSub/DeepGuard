const { createProxyMiddleware } = require('http-proxy-middleware');

module.exports = (app) => {
  const proxy = createProxyMiddleware({
    target: 'http://localhost:8000',
    changeOrigin: true,
  });

  app.use('/api', proxy);                          // /api/* → localhost:8000/api/*
  app.use(['/static/uploads', '/static/explain'], proxy); 
};