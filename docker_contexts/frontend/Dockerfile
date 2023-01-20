# Frontend reactjs
# Use whatever version you are running locally (see node -v)
FROM node:18

WORKDIR /app

# Install dependencies (you are already in /app)
COPY pcld-app/package.json pcld-app/package-lock.json .
RUN npm ci

# Add rest of the client code
# .dockerignore needs to skip node_modules
COPY pcld-app/. .

EXPOSE 3000

CMD bash
CMD ["npm", "start"]
