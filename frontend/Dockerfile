# React
FROM node:20-alpine AS build
WORKDIR /app
COPY package.json package-lock.json* ./
RUN npm ci
COPY . .
RUN npm run build

# serve with nginx
FROM nginx:stable-alpine
COPY --from=build /app/build /usr/share/nginx/html
# Proxy API requests to backend service
RUN sed -i 's|location / {|location / { proxy_pass http://backend:5000; }|' /etc/nginx/conf.d/default.conf
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
