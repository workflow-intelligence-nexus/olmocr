# Docker Package Compare MCP Server

A Model Context Protocol (MCP) server for comparing package versions between Docker containers and local environments.

## Features

- Compare Python versions between Docker and local
- Compare CUDA toolkit versions
- Analyze package version differences
- Support for conda and virtualenv environments
- Docker Compose configuration validation

## Installation

### Local Installation
```bash
git clone https://github.com/your-repo/docker-pkg-compare-mcp.git
cd docker-pkg-compare-mcp
pip install -r requirements.txt
```

### Docker Installation
```bash
docker build -t docker-pkg-compare-mcp .
```

## Usage

### Local Usage
```bash
python src/server.py
```

### Docker Usage
```bash
docker run -p 8000:8000 docker-pkg-compare-mcp
```

### Docker Compose
```yaml
version: '3'
services:
  pkg-compare:
    image: docker-pkg-compare-mcp
    ports:
      - "8000:8000"
    volumes:
      - /var/run/docker.sock:/var/run/docker.sock
```

Then connect to it using any MCP client. The available tool is:

- `compare_environments`: Compare Docker and local environments

Example request:
```json
{
  "dockerfile_path": "/path/to/Dockerfile",
  "requirements_path": "/path/to/requirements.txt",
  "compose_path": "/path/to/docker-compose.yml",
  "conda_env": "my-env"
}
```

## Configuration

The server can be configured via environment variables:

- `PORT`: Server port (default: 8000)
- `HOST`: Server host (default: 0.0.0.0)

## Integration with Cline and Windsurf

### Cline Integration
1. Start the MCP server (either locally or in Docker)
2. In Cline, add the server connection:
```bash
cline mcp add docker-pkg-compare http://localhost:8000
```
3. Verify connection:
```bash
cline mcp list
```
4. Use the comparison tool:
```bash
cline mcp exec docker-pkg-compare compare_environments --dockerfile_path ./Dockerfile --conda_env my-env
```

### Windsurf Integration
1. Add the MCP server to your Windsurf configuration:
```yaml
mcp_servers:
  docker-pkg-compare:
    url: http://localhost:8000
    tools:
      - compare_environments
```
2. Use in your workflow:
```yaml
steps:
  - name: Compare environments
    uses: mcp/docker-pkg-compare/compare_environments
    with:
      dockerfile_path: ./Dockerfile
      conda_env: my-env
```

For both tools, you can access the full comparison report through their respective UIs.
