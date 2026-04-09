export class SwarmCanvas {
  constructor(canvas, world) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
    this.world = world;
  }

  setWorld(world) {
    this.world = world;
  }

  worldToCanvas(x, y, viewport = { x: 0, y: 0, width: this.canvas.width, height: this.canvas.height }) {
    return {
      x: viewport.x + (x / this.world.width) * viewport.width,
      y: viewport.y + (y / this.world.height) * viewport.height,
    };
  }

  drawGrid(viewport = { x: 0, y: 0, width: this.canvas.width, height: this.canvas.height }) {
    const { ctx, canvas } = this;
    ctx.save();
    ctx.strokeStyle = "rgba(145, 200, 220, 0.08)";
    ctx.lineWidth = 1;

    const step = 50;
    for (let x = viewport.x; x <= viewport.x + viewport.width; x += step) {
      ctx.beginPath();
      ctx.moveTo(x, viewport.y);
      ctx.lineTo(x, viewport.y + viewport.height);
      ctx.stroke();
    }
    for (let y = viewport.y; y <= viewport.y + viewport.height; y += step) {
      ctx.beginPath();
      ctx.moveTo(viewport.x, y);
      ctx.lineTo(viewport.x + viewport.width, y);
      ctx.stroke();
    }
    ctx.restore();
  }

  drawLinks(links, agentsById, viewport) {
    const { ctx } = this;
    ctx.save();
    ctx.strokeStyle = "rgba(78, 205, 196, 0.22)";
    ctx.lineWidth = 1.2;
    for (const [leftId, rightId] of links) {
      const left = agentsById.get(leftId);
      const right = agentsById.get(rightId);
      if (!left || !right || !left.alive || !right.alive) {
        continue;
      }
      const p1 = this.worldToCanvas(left.x, left.y, viewport);
      const p2 = this.worldToCanvas(right.x, right.y, viewport);
      ctx.beginPath();
      ctx.moveTo(p1.x, p1.y);
      ctx.lineTo(p2.x, p2.y);
      ctx.stroke();
    }
    ctx.restore();
  }

  drawTargets(targets, viewport) {
    const { ctx } = this;
    ctx.save();

    for (const target of targets) {
      const p = this.worldToCanvas(target.x, target.y, viewport);
      const color =
        target.className === "vehicle"
          ? "#f4b942"
          : target.className === "decoy"
            ? "#ff6b6b"
            : "#8aa7b5";

      ctx.strokeStyle = color;
      ctx.fillStyle = target.active ? "rgba(244, 185, 66, 0.08)" : "rgba(120, 150, 160, 0.06)";
      ctx.beginPath();
      ctx.arc(p.x, p.y, target.active ? 14 : 10, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();

      ctx.fillStyle = "#d8edf6";
      ctx.font = "12px IBM Plex Mono";
      ctx.fillText(`T${target.id}`, p.x + 10, p.y - 10);
    }

    ctx.restore();
  }

  drawAgents(agents, viewport) {
    const { ctx } = this;
    ctx.save();

    for (const agent of agents) {
      const p = this.worldToCanvas(agent.x, agent.y, viewport);
      const heading = Math.atan2(agent.vy, agent.vx);

      ctx.translate(p.x, p.y);
      ctx.rotate(heading);

      ctx.fillStyle = agent.alive ? "#4ecdc4" : "#596c75";
      ctx.strokeStyle = "rgba(230, 245, 251, 0.8)";
      ctx.lineWidth = 1;

      ctx.beginPath();
      ctx.moveTo(9, 0);
      ctx.lineTo(-7, -5);
      ctx.lineTo(-5, 0);
      ctx.lineTo(-7, 5);
      ctx.closePath();
      ctx.fill();
      ctx.stroke();

      ctx.resetTransform();
    }

    ctx.restore();
  }

  drawHud(frame, viewport, label) {
    const { ctx, canvas } = this;
    ctx.save();
    ctx.fillStyle = "rgba(11, 30, 42, 0.78)";
    ctx.fillRect(viewport.x + 10, viewport.y + 10, 220, 56);
    ctx.strokeStyle = "rgba(147, 206, 229, 0.3)";
    ctx.strokeRect(viewport.x + 10, viewport.y + 10, 220, 56);

    ctx.fillStyle = "#9cc6d7";
    ctx.font = "12px IBM Plex Mono";
    ctx.fillText(label, viewport.x + 20, viewport.y + 28);
    ctx.fillText(`step: ${frame.t}`, viewport.x + 20, viewport.y + 46);
    ctx.fillText(`events: ${frame.events.length}`, viewport.x + 120, viewport.y + 46);

    if (frame.events[0]) {
      ctx.fillStyle = "rgba(244, 185, 66, 0.95)";
      ctx.fillText(frame.events[0].slice(0, 34), viewport.x + 20, viewport.y + viewport.height - 18);
    }

    ctx.restore();
  }

  drawLegend(viewport) {
    const { ctx } = this;
    const left = viewport.x + viewport.width - 220;
    const top = viewport.y + 10;

    ctx.save();
    ctx.fillStyle = "rgba(11, 30, 42, 0.78)";
    ctx.fillRect(left, top, 210, 112);
    ctx.strokeStyle = "rgba(147, 206, 229, 0.3)";
    ctx.strokeRect(left, top, 210, 112);

    ctx.font = "11px IBM Plex Mono";
    ctx.fillStyle = "#9cc6d7";
    ctx.fillText("图例", left + 10, top + 16);

    const rows = [
      ["#4ecdc4", "Agent"],
      ["#f4b942", "Vehicle Target"],
      ["#ff6b6b", "Decoy Target"],
      ["#8aa7b5", "Civilian Target"],
      ["#596c75", "Fail-safe Agent"],
    ];
    rows.forEach((row, idx) => {
      const y = top + 30 + idx * 16;
      ctx.fillStyle = row[0];
      ctx.beginPath();
      ctx.arc(left + 12, y - 3, 4, 0, Math.PI * 2);
      ctx.fill();
      ctx.fillStyle = "#d5ecf6";
      ctx.fillText(row[1], left + 24, y);
    });

    ctx.strokeStyle = "rgba(78, 205, 196, 0.65)";
    ctx.beginPath();
    ctx.moveTo(left + 10, top + 106);
    ctx.lineTo(left + 28, top + 106);
    ctx.stroke();
    ctx.fillStyle = "#d5ecf6";
    ctx.fillText("通信链路", left + 34, top + 109);
    ctx.restore();
  }

  drawViewport(frame, viewport, label) {
    const { ctx } = this;
    const agentsById = new Map(frame.agents.map((agent) => [agent.id, agent]));
    ctx.save();
    ctx.beginPath();
    ctx.rect(viewport.x, viewport.y, viewport.width, viewport.height);
    ctx.clip();
    this.drawGrid(viewport);
    this.drawLinks(frame.links, agentsById, viewport);
    this.drawTargets(frame.targets, viewport);
    this.drawAgents(frame.agents, viewport);
    this.drawHud(frame, viewport, label);
    this.drawLegend(viewport);
    ctx.restore();
  }

  render(frame) {
    const { ctx, canvas } = this;

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    this.drawViewport(frame, { x: 0, y: 0, width: canvas.width, height: canvas.height }, "单视图");
  }

  renderCompare(primaryFrame, secondaryFrame, primaryLabel, secondaryLabel) {
    const { ctx, canvas } = this;
    const gutter = 12;
    const halfWidth = (canvas.width - gutter) / 2;
    const leftViewport = { x: 0, y: 0, width: halfWidth, height: canvas.height };
    const rightViewport = { x: halfWidth + gutter, y: 0, width: halfWidth, height: canvas.height };

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    this.drawViewport(primaryFrame, leftViewport, primaryLabel);
    this.drawViewport(secondaryFrame, rightViewport, secondaryLabel);

    ctx.save();
    ctx.strokeStyle = "rgba(147, 206, 229, 0.18)";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(halfWidth + gutter / 2, 0);
    ctx.lineTo(halfWidth + gutter / 2, canvas.height);
    ctx.stroke();
    ctx.restore();
  }
}
