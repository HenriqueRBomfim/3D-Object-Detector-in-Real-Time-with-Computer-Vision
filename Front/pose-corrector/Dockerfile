FROM node:18-alpine
WORKDIR /app
COPY package*.json ./
RUN npm install
COPY . .

# não precisa de build em modo dev
EXPOSE 3000
CMD ["npm", "run", "dev"]