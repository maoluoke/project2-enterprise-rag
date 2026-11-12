"""Skill 装载器：扫描 skills/ 目录，提供技能元数据索引与按需加载。

目录约定（Anthropic Agent Skills 风格）：
    skills/
    └── <skill-name>/
        └── SKILL.md      # 首行 YAML front matter：name / description，正文为指令

工作机制：
- list_skills() 只读取各 SKILL.md 的 front matter（名称+描述），
  用于注入 system prompt，开销极小；
- load_skill(name) 才读取完整正文，供 Agent 在判断需要时按需拉取。
"""

import os
import re

SKILLS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "skills")

_FRONT_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n?(.*)$", re.DOTALL)


def _parse_skill_md(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        raw = f.read()
    m = _FRONT_RE.match(raw)
    meta = {"name": "", "description": ""}
    body = raw
    if m:
        for line in m.group(1).splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                meta[k.strip()] = v.strip().strip("\"'")
        body = m.group(2).strip()
    return {"meta": meta, "body": body}


def list_skills() -> list:
    """返回所有技能的元数据列表 [{name, description}]。"""
    skills = []
    if not os.path.isdir(SKILLS_DIR):
        return skills
    for entry in sorted(os.listdir(SKILLS_DIR)):
        md = os.path.join(SKILLS_DIR, entry, "SKILL.md")
        if os.path.isfile(md):
            parsed = _parse_skill_md(md)
            skills.append({
                "name": parsed["meta"].get("name") or entry,
                "description": parsed["meta"].get("description", ""),
            })
    return skills


def load_skill(name: str) -> str:
    """按名称加载技能完整指令正文；找不到时返回错误说明。"""
    md = os.path.join(SKILLS_DIR, name, "SKILL.md")
    if not os.path.isfile(md):
        available = [s["name"] for s in list_skills()]
        return f"Skill '{name}' not found. Available skills: {available}"
    return _parse_skill_md(md)["body"]


def skills_prompt_section() -> str:
    """生成注入 system prompt 的技能索引段落。"""
    skills = list_skills()
    if not skills:
        return ""
    lines = ["Available skills (call load_skill to fetch full instructions when relevant):"]
    for s in skills:
        lines.append(f"- {s['name']}: {s['description']}")
    return "\n".join(lines)
