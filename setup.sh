# Pull Ollama models
ollama pull llama3.2:3b
ollama pull qwen3:4b
ollama pull gemma3:4b
ollama pull phi4-mini:3.8b

# Caution: Terminate existing ports
lsof -i tcp:8080 | awk -F" " '{ print "kill " $2 }' > close_port.sh ; sh close_port.sh; rm close_port.sh

# Docker pull to deploy HAPI FHIR app
docker pull hapiproject/hapi:latest
docker run -p 8080:8080 hapiproject/hapi:latest

