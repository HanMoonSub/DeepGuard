FROM redis:8.10-alpine

COPY docker/redis/config/redis.conf /usr/local/etc/redis/redis.conf

ENV TZ=Asia/Seoul

EXPOSE 6379

CMD ["redis-server", "/usr/local/etc/redis/redis.conf"]
