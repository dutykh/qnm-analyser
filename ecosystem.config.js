// QNM Analyser — PM2 process definitions
// Author: Dr. Denys Dutykh (https://www.denys-dutykh.com/)
//
//   pm2 start ecosystem.config.js
//   pm2 save
//
// Both processes bind loopback only; Traefik is the public entry point.

module.exports = {
  apps: [
    {
      name: 'qnm-analyser',
      script: './start.sh',
      cwd: __dirname,
      interpreter: 'bash',
      autorestart: true,
      max_restarts: 10,
      min_uptime: '20s',
      // Gunicorn already recycles workers; this is a backstop for the master.
      max_memory_restart: '1G',
      kill_timeout: 10000,
      env: {
        OMP_NUM_THREADS: '1',
        OPENBLAS_NUM_THREADS: '1',
        MKL_NUM_THREADS: '1',
        NUMEXPR_NUM_THREADS: '1',
      },
    },
    {
      name: 'qnm-webhook',
      script: './webhook_start.sh',
      cwd: __dirname,
      interpreter: 'bash',
      autorestart: true,
      max_restarts: 10,
      min_uptime: '20s',
      max_memory_restart: '256M',
      kill_timeout: 10000,
    },
  ],
};
