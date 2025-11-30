# frontend/Dockerfile

FROM node:20-alpine

WORKDIR /app

COPY package.json package-lock.json* pnpm-lock.yaml* yarn.lock* ./
# If you use npm, the above is okay. It will copy whichever lockfile exists.

RUN npm install

COPY . .

EXPOSE 3000

CMD ["npm", "run", "dev"]
