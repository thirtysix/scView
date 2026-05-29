FROM node:20-slim AS base
WORKDIR /app
COPY package.json package-lock.json* ./

# Dev target
FROM base AS dev
RUN npm install
COPY . .
EXPOSE 5173
CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0", "--port", "5173"]

# Build target
FROM base AS build
RUN npm install
COPY . .
RUN npm run build

# Production target
FROM nginx:alpine AS production
COPY --from=build /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
