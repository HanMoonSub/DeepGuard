FROM node:26.8-alpine AS build

WORKDIR /app

COPY package.json package-lock.json ./
RUN npm ci

COPY . .
ENV GENERATE_SOURCEMAP=false
RUN npm run build

FROM nginx:1.31-alpine

COPY --from=build /app/build /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf

ENV TZ=Asia/Seoul

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
