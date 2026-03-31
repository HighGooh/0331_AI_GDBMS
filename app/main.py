import ollama
import json
import re
import os
from typing import List, Optional, Dict, Union
from requests import get
from bs4 import BeautifulSoup as bs
from pydantic import BaseModel

# 속성 타입 정의 (문자열, 숫자, bool, None)
PropertyValue = Union[str, int, float, bool, None]

# ---------------------------
# 지식 그래프 기본 모델 정의
# ---------------------------
class Node(BaseModel):
  id: str  # 노드 ID (예: N0)
  label: str  # 노드 타입 (예: "인간")
  properties: Optional[Dict[str, PropertyValue]] = None  # 속성 딕셔너리

class Relationship(BaseModel):
  type: str  # 관계 유형
  start_node_id: str  # 시작 노드 ID
  end_node_id: str  # 끝 노드 ID
  properties: Optional[Dict[str, PropertyValue]] = None  # 관계 속성

class GraphResponse(BaseModel):
  nodes: List[Node]  # 노드 리스트
  relationships: List[Relationship]  # 관계 리스트

# ----------------------------------------
# LLM에 전달되는 템플릿: 노드와 관계 추출 규칙
# ----------------------------------------
UPDATED_TEMPLATE = """
### Role
You are a high-precision Knowledge Graph engineer specialized in structured JSON output.

### Extraction Task
Extract entities and relationships from the provided synopsis.
- Use ONLY the provided `NODES` list for IDs.
- If a character/Pokémon is not in the `NODES` list, SKIP it entirely.

### 🚨 STRICT OUTPUT FORMAT (MANDATORY)
Return a SINGLE JSON object with this exact structure. Do NOT include markdown code blocks or any text outside the JSON.

{
  "nodes": [
    {
      "id": "N0",
      "label": "인간",
      "properties": {"name": "Ash Ketchum"}
    }
  ],
  "relationships": [
    {
      "type": "TRAVELS_WITH",
      "start_node_id": "N0",
      "end_node_id": "N1",
      "properties": {
        "episode_number": "S02E01",
        "context": "Brief reason for this relationship in this episode"
      }
    }
  ]
}

### 🚨 NODE & ID RULES
1. **NO NEW IDs:** Do not create IDs like "N101" or "Temp1". If not in the list, ignore the entity.
2. **ID LOOKUP:** - Ash Ketchum/지우 -> N0
   - Misty/이슬 -> N1
   - Pikachu/피카츄 -> N21
   - (Check the provided NODES list carefully before assigning an ID).
3. **NO DUPLICATES:** Include a node or relationship only once per response.

### 🚨 RELATIONSHIP RULES
1. **REQUIRED FIELD:** Every relationship object MUST contain the "type" key.
2. **TYPE LIST:** Use ONLY these types: [OWNS, CAUGHT, TRAVELS_WITH, WORKS_FOR, EVOLVES_FROM, BATTLES, MEETS].
3. **TEAM ROCKET LOGIC:** Map "Team Rocket" to Jessie (N5), James (N6), and Meowth (N22). Link all three to Giovanni (N14) via `WORKS_FOR`.

NODES = [
  {"id": "N0", "label": "인간", "properties": {"name": "Ash Ketchum"}},
  {"id": "N1", "label": "인간", "properties": {"name": "Misty"}},
  {"id": "N2", "label": "인간", "properties": {"name": "Brock"}},
  {"id": "N3", "label": "인간", "properties": {"name": "Professor Oak"}},
  {"id": "N4", "label": "인간", "properties": {"name": "Gary Oak"}},
  {"id": "N5", "label": "인간", "properties": {"name": "Jessie"}},
  {"id": "N6", "label": "인간", "properties": {"name": "James"}},
  {"id": "N7", "label": "인간", "properties": {"name": "Officer Jenny"}},
  {"id": "N8", "label": "인간", "properties": {"name": "Nurse Joy"}},
  {"id": "N9", "label": "인간", "properties": {"name": "Lt. Surge"}},
  {"id": "N10", "label": "인간", "properties": {"name": "Bill"}},
  {"id": "N11", "label": "인간", "properties": {"name": "Sabrina"}},
  {"id": "N12", "label": "인간", "properties": {"name": "Erika"}},
  {"id": "N13", "label": "인간", "properties": {"name": "Koga"}},
  {"id": "N14", "label": "인간", "properties": {"name": "Giovanni"}},
  {"id": "N15", "label": "인간", "properties": {"name": "Blaine"}},
  {"id": "N16", "label": "인간", "properties": {"name": "Delia Ketchum"}},
  {"id": "N17", "label": "인간", "properties": {"name": "Bruno"}},
  {"id": "N18", "label": "인간", "properties": {"name": "Richie"}},
  {"id": "N19", "label": "인간", "properties": {"name": "Charles Goodshow"}},
  {"id": "N20", "label": "인간", "properties": {"name": "Professor Ivy"}},
  {"id": "N21", "label": "포켓몬", "properties": {"name": "Pikachu"}},
  {"id": "N22", "label": "포켓몬", "properties": {"name": "Meowth"}},
  {"id": "N23", "label": "포켓몬", "properties": {"name": "Spearow"}},
  {"id": "N24", "label": "포켓몬", "properties": {"name": "Ho-Oh"}},
  {"id": "N25", "label": "포켓몬", "properties": {"name": "Caterpie"}},
  {"id": "N26", "label": "포켓몬", "properties": {"name": "Metapod"}},
  {"id": "N27", "label": "포켓몬", "properties": {"name": "Butterfree"}},
  {"id": "N28", "label": "포켓몬", "properties": {"name": "Pidgeotto"}},
  {"id": "N29", "label": "포켓몬", "properties": {"name": "Ekans"}},
  {"id": "N30", "label": "포켓몬", "properties": {"name": "Koffing"}},
  {"id": "N31", "label": "포켓몬", "properties": {"name": "Raichu"}},
  {"id": "N32", "label": "포켓몬", "properties": {"name": "Bulbasaur"}},
  {"id": "N33", "label": "포켓몬", "properties": {"name": "Charmander"}},
  {"id": "N34", "label": "포켓몬", "properties": {"name": "Squirtle"}},
  {"id": "N35", "label": "포켓몬", "properties": {"name": "Dragonite"}},
  {"id": "N36", "label": "포켓몬", "properties": {"name": "Staryu"}},
  {"id": "N37", "label": "포켓몬", "properties": {"name": "Starmie"}},
  {"id": "N38", "label": "포켓몬", "properties": {"name": "Onix"}},
  {"id": "N39", "label": "포켓몬", "properties": {"name": "Geodude"}},
  {"id": "N40", "label": "포켓몬", "properties": {"name": "Zubat"}},
  {"id": "N41", "label": "포켓몬", "properties": {"name": "Abra"}},
  {"id": "N42", "label": "포켓몬", "properties": {"name": "Kadabra"}},
  {"id": "N43", "label": "포켓몬", "properties": {"name": "Haunter"}},
  {"id": "N44", "label": "포켓몬", "properties": {"name": "Gastly"}},
  {"id": "N45", "label": "포켓몬", "properties": {"name": "Gengar"}},
  {"id": "N46", "label": "포켓몬", "properties": {"name": "Gloom"}},
  {"id": "N47", "label": "포켓몬", "properties": {"name": "Primeape"}},
  {"id": "N48", "label": "포켓몬", "properties": {"name": "Muk"}},
  {"id": "N49", "label": "포켓몬", "properties": {"name": "Diglett"}},
  {"id": "N50", "label": "포켓몬", "properties": {"name": "Dugtrio"}},
  {"id": "N51", "label": "포켓몬", "properties": {"name": "Venomoth"}},
  {"id": "N52", "label": "포켓몬", "properties": {"name": "Ponyta"}},
  {"id": "N53", "label": "포켓몬", "properties": {"name": "Rapidash"}},
  {"id": "N54", "label": "포켓몬", "properties": {"name": "Dratini"}},
  {"id": "N55", "label": "포켓몬", "properties": {"name": "Dragonair"}},
  {"id": "N56", "label": "포켓몬", "properties": {"name": "Tauros"}},
  {"id": "N57", "label": "포켓몬", "properties": {"name": "Ditto"}},
  {"id": "N58", "label": "포켓몬", "properties": {"name": "Eevee"}},
  {"id": "N59", "label": "포켓몬", "properties": {"name": "Snorlax"}},
  {"id": "N60", "label": "포켓몬", "properties": {"name": "Scyther"}},
  {"id": "N61", "label": "포켓몬", "properties": {"name": "Electabuzz"}},
  {"id": "N62", "label": "포켓몬", "properties": {"name": "Magmar"}},
  {"id": "N63", "label": "포켓몬", "properties": {"name": "Jigglypuff"}},
  {"id": "N64", "label": "포켓몬", "properties": {"name": "Aerodactyl"}},
  {"id": "N65", "label": "포켓몬", "properties": {"name": "Togepi"}},
  {"id": "N66", "label": "포켓몬", "properties": {"name": "Mewtwo"}},
  {"id": "N67", "label": "포켓몬", "properties": {"name": "Mr. Mime"}},
  {"id": "N68", "label": "포켓몬", "properties": {"name": "Lapras"}},
  {"id": "N69", "label": "포켓몬", "properties": {"name": "Sparky"}},
  {"id": "N70", "label": "포켓몬", "properties": {"name": "Beedrill"}},
  {"id": "N71", "label": "포켓몬", "properties": {"name": "Krabby"}},
  {"id": "N72", "label": "포켓몬", "properties": {"name": "Kingler"}},
  {"id": "N73", "label": "인간", "properties": {"name": "Kaoruko"}},
  {"id": "N74", "label": "포켓몬", "properties": {"name": "Golem"}},
  {"id": "N75", "label": "인간", "properties": {"name": "Melissa"}},
  {"id": "N76", "label": "인간", "properties": {"name": "Mandi"}},
  {"id": "N77", "label": "인간", "properties": {"name": "Ritchie"}},
  {"id": "N78", "label": "포켓몬", "properties": {"name": "Nidoking"}},
  {"id": "N79", "label": "포켓몬", "properties": {"name": "Bellsprout"}},
  {"id": "N80", "label": "포켓몬", "properties": {"name": "Charizard"}},
  {"id": "N81", "label": "포켓몬", "properties": {"name": "Shellder"}},
  {"id": "N82", "label": "포켓몬", "properties": {"name": "Psyduck"}},
  {"id": "N83", "label": "포켓몬", "properties": {"name": "Parasect"}},
  {"id": "N84", "label": "포켓몬", "properties": {"name": "Blastoise"}},
  {"id": "N85", "label": "포켓몬", "properties": {"name": "Slowbro"}},
  {"id": "N86", "label": "인간", "properties": {"name": "Cassandra"}},
  {"id": "N87", "label": "포켓몬", "properties": {"name": "Paras"}},
  {"id": "N88", "label": "포켓몬", "properties": {"name": "Slowpoke"}},
  {"id": "N89", "label": "포켓몬", "properties": {"name": "Wartortle"}},
  {"id": "N90", "label": "인간", "properties": {"name": "Professor Westwood V"}},
  {"id": "N91", "label": "포켓몬", "properties": {"name": "Arcanine"}},
  {"id": "N92", "label": "포켓몬", "properties": {"name": "Ivysaur"}},
  {"id": "N93", "label": "포켓몬", "properties": {"name": "Ninetales"}},
  {"id": "N94", "label": "포켓몬", "properties": {"name": "Moltres"}},
  {"id": "N95", "label": "포켓몬", "properties": {"name": "Danny"}},
  {"id": "N96", "label": "인간", "properties": {"name": "Seymour"}},
  {"id": "N97", "label": "포켓몬", "properties": {"name": "Alakazam"}},
  {"id": "N98", "label": "인간", "properties": {"name": "Tommy's Parents"}},
  {"id": "N99", "label": "포켓몬", "properties": {"name": "Weezing"}},
  {"id": "N100", "label": "포켓몬", "properties": {"name": "Cloyster"}},
  {"id": "N101", "label": "인간", "properties": {"name": "Duplica"}},
  {"id": "N102", "label": "포켓몬", "properties": {"name": "Ditto"}},
  {"id": "N103", "label": "인간", "properties": {"name": "Damian"}}, # 파이리 버린 트레이너
  {"id": "N104", "label": "인간", "properties": {"name": "Aya"}},    # 독수 여동생
  {"id": "N105", "label": "인간", "properties": {"name": "Jeanette Fisher"}}, # 4차전 라이벌
  {"id": "N106", "label": "포켓몬", "properties": {"name": "Victreebel"}},
  {"id": "N107", "label": "인간", "properties": {"name": "Butch"}},   # 로켓단 코산
  {"id": "N108", "label": "인간", "properties": {"name": "Cassidy"}}, # 로켓단 코사
  {"id": "N109", "label": "인간", "properties": {"name": "Tracey Sketchit"}}, # 관철 (웅이 대신 합류)
  {"id": "N110", "label": "포켓몬", "properties": {"name": "Marill"}},      # 관철의 포켓몬
  {"id": "N111", "label": "포켓몬", "properties": {"name": "Venonat"}},     # 관철의 포켓몬
  {"id": "N112", "label": "인간", "properties": {"name": "Cissy"}},        # 강미 (첫 번째 관장)
  {"id": "N113", "label": "인간", "properties": {"name": "Rudy"}},         # 지코 (세 번째 관장)
  {"id": "N114", "label": "인간", "properties": {"name": "Luana"}},        # 루리 (네 번째 관장)
  {"id": "N115", "label": "인간", "properties": {"name": "Drake"}},        # 강산 (헤드 트레이너)
  {"id": "N116", "label": "포켓몬", "properties": {"name": "Crystal Onix"}}, # 수정 롱스톤
  {"id": "N117", "label": "포켓몬", "properties": {"name": "Politeod"}},
  {"id": "N118", "label": "포켓몬", "properties": {"name": "Poliwag"}},     # 이슬의 발챙이
  {"id": "N119", "label": "포켓몬", "properties": {"name": "Scyther"}}      # 관철의 스라크
]
"""

# 영어 이름 → 한글 이름 변환 매핑 테이블
KOREAN_NODE_MAP = {
    "Ash Ketchum": "지우",
    "Pikachu": "피카츄",
    "Misty": "이슬",
    "Brock": "웅",
    "Professor Oak": "오박사",
    "Gary Oak": "바람",
    "Jessie": "로사",
    "James": "로이",
    "Meowth": "나옹",
    "Spearow": "깨비참",
    "Ho-Oh": "칠색조",
    "Officer Jenny": "여경",
    "Nurse Joy": "간호순",
    "Caterpie": "캐터피",
    "Metapod": "단데기",
    "Butterfree": "버터플",
    "Pidgeotto": "피죤",
    "Ekans": "아보",
    "Koffing": "또가스",
    "Lt. Surge": "마티스",
    "Raichu": "라이츄",
    "Bulbasaur": "이상해씨",
    "Charmander": "파이리",
    "Squirtle": "꼬부기",
    "Bill": "이수재",
    "Dragonite": "망나뇽",
    "Staryu": "별가사리",
    "Starmie": "아쿠스타",
    "Onix": "롱스톤",
    "Geodude": "꼬마돌",
    "Zubat": "주뱃",
    "Sabrina": "초련",
    "Abra": "캐이시",
    "Kadabra": "윤겔라",
    "Haunter": "고우스트",
    "Gastly": "고스",
    "Gengar": "팬텀",
    "Erika": "민화",
    "Gloom": "냄새꼬",
    "Primeape": "성원숭",
    "Muk": "질뻐기",
    "Diglett": "디그다",
    "Dugtrio": "닥트리오",
    "Koga": "독수",
    "Venomoth": "도나리",
    "Ponyta": "포니타",
    "Rapidash": "날쌩마",
    "Dratini": "미뇽",
    "Dragonair": "신용",
    "Tauros": "켄타로스",
    "Ditto": "메타몽",
    "Eevee": "이브이",
    "Snorlax": "잠만보",
    "Scyther": "스라크",
    "Electabuzz": "에레브",
    "Magmar": "마그마",
    "Jigglypuff": "푸린",
    "Aerodactyl": "프테라",
    "Togepi": "토게피",
    "Giovanni": "비주기",
    "Blaine": "강연",
    "Mewtwo": "뮤츠",
    "Mr. Mime": "마임맨",
    "Delia Ketchum": "영자",
    "Lapras": "라프라스",
    "Bruno": "시바",
    "Richie": "훈이",
    "Sparky": "레온",
    "Charles Goodshow": "리그 의장",
    "Professor Ivy": "미지박사",
    "Beedrill": "독침봉",
    "Krabby": "크랩",
    "Kingler": "킹크랩",
    "Kaoruko": "카오루코",
    "Golem": "딱구리",
    "Melissa": "멜리사",
    "Mandi": "재영",
    "Ritchie": "훈이",
    "Nidoking": "니드킹",
    "Bellsprout": "모다피",
    "Charizard": "리자몽",
    "Shellder": "셀러",
    "Psyduck": "고라파덕",
    "Parasect": "파라섹트",
    "Blastoise": "거북왕",
    "Slowbro": "야도란",
    "Cassandra": "링링",
    "Paras": "파라스",
    "Slowpoke": "야돈",
    "Wartortle": "어니부기",
    "Professor Westwood V": "포만물 박사",
    "Arcanine": "윈디",
    "Ivysaur": "이상해풀",
    "Ninetales": "나인테일",
    "Moltres" : "파이어",
    "Danny": "호남",
    "Seymour": "나해박",
    "Alakazam" : "후딘",
    "Tommy's Parents" : "다잔의 부모",
    "Weezing": "또도가스",
    "Cloyster": "파르셀",
    "Duplica": "희나",
    "Damian": "다솜",
    "Aya": "아야",
    "Jeanette Fisher": "훈희",
    "Butch": "코산",
    "Cassidy": "코사",
    "Tracey Sketchit": "관철",
    "Cissy": "강미",
    "Rudy": "지코",
    "Luana": "루리",
    "Drake": "강산",
    "Ditto": "메타몽",
    "Victreebel": "우츠보트",
    "Marill": "마릴",
    "Venonat": "콘팡",
    "Crystal Onix": "크리스탈 롱스톤",
    "Politeod": "왕구리",
    "Poliwag": "발챙이",
    "Scyther": "스라크"
}

# ---------------------------
# Ollama LLM 호출 함수
# ---------------------------
def llm_call_structured(prompt: str, model: str = "Llama3.1:8B") -> GraphResponse:

  final_prompt = prompt + """
  Return ONLY valid JSON. Do NOT include explanations or commentary.
  """

  # Ollama에 LLM 요청
  response = ollama.chat(
    model=model,
    messages=[{"role": "user", "content": final_prompt}]
  )

  # 모델 응답 텍스트 추출
  text = response["message"]["content"]

  # JSON 파싱 시도
  try:
    parsed = json.loads(text)
  except json.JSONDecodeError:
    # 전체 텍스트에서 JSON 블록만 추출
    json_text = re.search(r"\{.*\}", text, re.S)
    if not json_text:
      raise Exception("모델 응답에서 JSON을 찾지 못했습니다.")
    parsed = json.loads(json_text.group(0))
  
  return GraphResponse(**parsed)  # pydantic 모델로 변환 후 반환

# ------------------------------------------------------
# 여러 에피소드 그래프를 통합하기 위한 함수
# ------------------------------------------------------
def combine_chunk_graphs(chunk_graphs: list) -> GraphResponse:
  all_nodes = []  # 모든 노드를 담을 리스트
  for chunk_graph in chunk_graphs:
    for node in chunk_graph.nodes:
      all_nodes.append(node)
  
  all_relationships = []  # 모든 관계를 담을 리스트
  for chunk_graph in chunk_graphs:
    for relationship in chunk_graph.relationships:
      all_relationships.append(relationship)
  
  unique_nodes = []  # 중복 제거된 최종 노드 리스트
  seen = set()  # 노드 중복 체크용

  for node in all_nodes:
    node_key = (node.id, node.label, str(node.properties))  # 노드 고유값 생성
    if node_key not in seen:
      unique_nodes.append(node)
      seen.add(node_key)

  return GraphResponse(nodes=unique_nodes, relationships=all_relationships)

# ------------------------------------------------------
# 수집된 데이터를 LLM으로 처리하여 그래프 생성
# ------------------------------------------------------
def process_data(episodes: List[dict]) -> GraphResponse:
  print("=== 데이터 처리 시작 ===")

  chunk_graphs: List[GraphResponse] = []  # 에피소드별 그래프 저장
    
  for episode in episodes:
    if not episode.get("synopsis"):
      print(f"에피소드 S{episode['season']}E{episode['episode_in_season']:02d}: 시놉시스가 없어 건너뜀")
      continue
        
    print(f"에피소드 처리 중: 시즌 {episode['season']}, 에피소드 {episode['episode_in_season']}")
    
    try:
      prompt = UPDATED_TEMPLATE + f"\n 입력값\n {episode['synopsis']}"  # LLM 입력 프롬프트
      graph_response = llm_call_structured(prompt)  # LLM 호출
      episode_number = f"S{episode['season']}E{episode['episode_in_season']:02d}"  # 에피소드 번호 문자열

      for relationship in graph_response.relationships:
        if relationship.properties is None:
          relationship.properties = {}
        relationship.properties["episode_number"] = episode_number  # 관계에 에피소드 번호 부여
          
      for node in graph_response.nodes:
        english_name = node.properties.get("name", "")
        if english_name in KOREAN_NODE_MAP:
          node.properties["name"] = KOREAN_NODE_MAP[english_name]  # 영어 → 한글 변환
      
      chunk_graphs.append(graph_response)  # 결과 저장
    except Exception as e:
      print(f"  - 에피소드 처리 중 오류 발생: {e}")
      continue
  
  if not chunk_graphs:
    raise Exception("그래프를 성공적으로 추출하지 못했습니다.")
  
  print(f"총 {len(chunk_graphs)}개 에피소드 처리 완료")
  return combine_chunk_graphs(chunk_graphs)  # 전체 그래프 통합

# ------------------------------------------------------
# 위키피디아 에피소드 데이터 수집
# ------------------------------------------------------
def fetch_episode(link: str) -> List[dict]:
  headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}  # 요청 헤더
  response = get(link, headers=headers)  # GET 요청
  soup = bs(response.text, "html.parser")  # HTML 파싱
  season = soup.select_one("td.infobox-subheader").get_text(strip=True)
  print(season)
  # season = int(re.search(r"season_(\d+)", link).group(1))  # 시즌 번호 추출
  print(f"Fetching Season {season} from: {link}")
  
  table = soup.select_one("table.wikitable.plainrowheaders.wikiepisodetable")  # 에피소드 테이블 찾기
  episodes = []
  rows = table.select("tr.vevent.module-episode-list-row")  # 각 에피소드 row

  for i, row in enumerate(rows, start=1):  # 에피소드 번호 생성
    synopsis = None
    synopsis_row = row.find_next_sibling("tr", class_="expand-child")  # 시놉시스 row 찾기
    if synopsis_row:
      synopsis_cell = synopsis_row.select_one("td.description div.shortSummaryText")
      synopsis = synopsis_cell.get_text(strip=True) if synopsis_cell else None

    episodes.append({
      "season": season,
      "episode_in_season": i,
      "synopsis": synopsis,
    })

    # if i == 10:
    #   break
  
  return episodes

# ------------------------------------------------------
# 출력 파일 저장
# ------------------------------------------------------

def save_output(episodes: List[dict], final_graph: GraphResponse):
  print("=== 결과 저장 ===")
  
  os.makedirs("output", exist_ok=True)  # output 폴더 생성
  
  with open("output/1_원본데이터.json", "w", encoding="utf-8") as f:
      json.dump(episodes, f, indent=2, ensure_ascii=False)
  print("원본 데이터 저장: output/1_원본데이터.json")
  
  with open("output/지식그래프_최종.json", "w", encoding="utf-8") as f:
      json.dump(final_graph.model_dump(), f, ensure_ascii=False, indent=2)
  print("최종 지식그래프 저장: output/지식그래프_최종.json")

# ------------------------------------------------------
# 메인 실행 함수
# ------------------------------------------------------
def main():
  try:
    episode_links = [
      # "https://en.wikipedia.org/wiki/Pok%C3%A9mon:_Indigo_League",
      "https://en.wikipedia.org/wiki/Pok%C3%A9mon:_Adventures_in_the_Orange_Islands"
    ]
    all_episodes = []
    for link in episode_links:
      try:
        episodes = fetch_episode(link)
        all_episodes.extend(episodes)
      except Exception as e:
        print(f"Error fetching data from {link}: {e}")
        continue
    print(f"총 {len(all_episodes)}개 에피소드 수집 완료")

    final_graph = process_data(all_episodes)

    save_output(episodes, final_graph)  # 결과 저장
        
    print("=" * 50)
    print("✅ 지식그래프 생성 완료!")
    print(f"📊 총 노드 수: {len(final_graph.nodes)}")
    print(f"🔗 총 관계 수: {len(final_graph.relationships)}")    
  except Exception as e:
    print(f"오류 발생: {e}")
    return 1
  return 0

# ------------------------------------------------------
# 프로그램 실행
# ------------------------------------------------------
if __name__ == "__main__":
  exit(main())
