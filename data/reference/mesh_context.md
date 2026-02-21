# The Mesh — World Reference for Prismatic Studios

## Setting
The year is 2075, 55 years after "The Collision" — a simultaneous global surge of ley lines that tore open dimensional rifts at intersections. The Bay Area was one of the hardest-hit regions, built atop five major ley convergence points. Earth merged with three other dimensions: The Lattice (crystalline/geometric), The Deep (bioluminescent/abyssal), and The Archive (ruined alien civilization).

## The Mesh
A hybrid network fusing the pre-Collision internet with ley energy, co-built by Technomancers, Dragons, and AIs. Available wirelessly everywhere. Powers consciousness transfer (cloning/respawn), communication, and commerce. Decentralized — nobody owns it, everybody uses it.

## Zones & Districts

### The Valley (San Jose) — Corporate Megacity
Districts: Spire District (upper), The Grid (mid-city), The Foundation (undercity), Southern Wall, Genysis Campus.
Aesthetic: towering corporate arcologies, perpetual rain, neon reflecting on wet streets, steam from vents.

### The Bastion (Oakland) — Cyberknight Fortress
Districts: The Citadel (hilltop HQ), Covenant Row (mid-city), Portside (waterfront trade), The Remembrance (ruins memorial), Diablo Road (eastern gate).
Aesthetic: fortress walls over industrial infrastructure, disciplined military order, reclaimed pre-Collision architecture.

### Thornhold (Marin County) — Dragon City
Districts: The Canopy (upper), The Roots (street level), The Mountain (Mt. Tamalpais), Bridgehead (south), Muir (western forest).
Aesthetic: organic integration with ancient forest, dragon-scale architecture, living buildings woven with ley energy.

### Driftwood (Sonoma Coast) — Barrens Settlement
Districts: The Shelf (residential), The Harbor (cove), The Relay (Mesh tower), The Tidepools (rocky shore), Inland Trail.
Aesthetic: salvaged materials, resourceful engineering, wild coastline meeting ley-fed wilderness.

### Old San Francisco — Rift Destination
Sub-zones: Perimeter, Financial District (Lattice-dominant), Golden Gate Park (Deep/Ley), Presidio (Archive), Rift Core.
Golden Gate Bridge half-standing, phasing between realities. Bay Bridge destroyed, twisted pylons visible from Bastion.

## Character Classes

### Juicer (Alchemist) — Resource: Energy
Trees: Accelerant, Mutagen, Apothecary. Key abilities: Adrenal Spike, Metabolic Overclock, Burnout Strike, Adaptive Carapace, Apex Predator, Combat Stim Injection, Chem Cloud, Synthesis Cascade.

### Cyberknight — Resource: Charge
Trees: Bastion, Templar, Sentinel. Key abilities: Mag-Lock Stance, Shield Matrix, Iron Curtain, Powered Sweep, Vibro Resonance, Judgment, Hardlight Ricochet, Aegis Storm.

### Street Samurai — Resource: Energy
Trees: Gunslinger, Operator, Heavy. Key abilities: Double Tap, Ricochet, Bullet Ballet, Overwatch Position, Killshot, Ghost Protocol, Suppressive Zone, Rolling Thunder.

### Channeler/Ley Weaver — Resource: Mana
Trees: Conduit, Warden, Rift Caller. Key abilities: Ley Bolt, Resonance Cascade, Dimensional Breach, Ley Shield, Stabilization Pulse, Open Rift, Ley Bond, Convergence of Pacts.

### Adept/Ki-Runner — Resource: Energy
Trees: Iron Body, Shadow Step, Living Weapon. Key abilities: Stone Palm, Resonant Counter, Adamantine Form, Shadow Step, Fade, Terminus, Ley Fist, Hundred Hands, Perfect Form.

### Meshrunner — Resource: Charge
Trees: Swarm, Heavy Metal, Ghost in the Machine. Key abilities: Deploy Scout Wing, Drone Barrage, Cascade Failure, Weapons Platform Mode, Titan Protocol, System Intrusion, Cyberware Shutdown, Puppet Master.

### Shaman — Resource: Mana
Patrons: Spider, Dog, Hawk. Key abilities: Ley Web, Cocooning, Mother of Spiders, Mending Howl, Alpha's Call, Talon Strike, Cry of the Hunt, Extinction Dive.

### Technomancer — Resource: Charge
Trees: Sprite Weaver, Gridrunner, Echo. Key abilities: Compile Sprite, Digital Evolution, Emergent Intelligence, Data Spike, Mesh Storm, Technomantic Singularity, Phase Echo, Echo Split, Convergence.

## Key NPCs
- **Director Lian Zhao** — CEO of Genysis Bioworks, Valley power broker
- **Marcus "Zero" Okafor** — Decker, information broker in The Foundation
- **Dr. Aris Tan** — Consciousness transfer researcher, Genysis
- **Councilor Venn** — Crucible Arms representative
- **Knight-Commander Adaeze Obi** — Founder of the Bastion Cyberknights
- **Captain Renn Vasquez** — Bastion external operations
- **Brother Tomas Guerrero** — Ley Artificer, Bastion
- **Sylithra** — Ancient dragon ruling Thornhold from Mt. Tamalpais
- **Harlan Voss** — Thornhold administrator
- **Kaia "Whisper" Nakamura** — Information broker, Thornhold underground
- **Elder Moss** — Shaman enclave leader, Muir
- **Elder Maren Solis** — Driftwood council leader
- **"Patch"** — Driftwood Technomancer
- **Old Ravi** — Driftwood defense elder
- **Luma** — Shaman apprentice, Driftwood

## Factions
- **Genysis Bioworks** — Cloning corporation, Valley-based
- **Crucible Arms** — Weapons/military contractor
- **Meridian Systems** — Mesh infrastructure corporation
- **The Foundation Collective** — Undercity informal alliance
- **Order of the Iron Veil** — Cyberknight order, Bastion
- **The Anchor Corps** — Rift stabilization specialists
- **Dragon's Court** — Thornhold administrative apparatus
- **The Thornguard** — Thornhold military
- **Whisper's Web** — Underground cooperative, Thornhold
- **The Muir Enclave** — Shaman community
- **Coastal Trade Network** — Multi-settlement alliance, Driftwood

## Tech Stack (Studio)
- **Client:** Bevy 0.18 (Rust), legacy Unreal Engine 5 (C++)
- **Game Server:** Go 1.22, custom ECS, 20-30Hz tick rate
- **Database:** CockroachDB (primary), Redis 7 (cache/session)
- **Backend:** Nakama 3.x (auth, chat, friends), NATS + JetStream (events)
- **Infrastructure:** Kubernetes + Agones, Terraform, GitHub Actions CI/CD
- **Contracts:** Protocol Buffers via Buf

## Game Mechanics
- **Essence:** Characters start at 6.0. Cyberware costs Essence. Magic weakens as Essence drops. Core tension between chrome and magic.
- **Ley Tides:** Ley energy ebbs/flows on cycles. High tides strengthen magic users, cause cyberware glitches, open temporary rifts.
- **Rifts:** Minor (2 ley lines, 5-person), Mid-Tier (3-4 lines, 10-person), Major Nexus (5+, 20-40 person raids). Lifecycle: Stabilize, Clear, Anchor, Maintain.
- **Cloning/Death:** Consciousness transfers through Mesh to nearest clone vat. Safe zones = quick respawn + clone sickness. Contested zones = corpse recovery window. Hardcore rifts = full drop.
- **Dimensional Materials:** Lattice (geometric, vibro-weapon ideal), Deep (regenerative, bioluminescent), Archive (alien engineering).

## Currency & Items
- **Nuyen** (primary currency), Credits (scrip)
- Weapons: Street Katana, Titanium Combat Fist, Ares Predator V, AK-97, Ingram Smartgun X, Defiance T-250, Hermetic Power Focus, Void Reaver
- Crafting professions: Cybertech, Weaponsmith, Ley Artificer, Chemist, Mesh Engineer, Armorer (2-profession limit)

## Art Direction
- **Cities:** Rain-soaked noir cyberpunk. Neon on wet streets, steam vents, vertical stratification, persistent rain/mist, moss and fungi glowing with ley energy.
- **Wilderness:** Explosion of life — dense jungle, ancient forest, overgrown ruins. Luminescent rivers, bioluminescent ley lines (blues, greens, golds).
- **Rifts:** Lattice = angular, refractive, prismatic. Deep = bioluminescent, membrane walls, pulsing floors. Archive = alien architecture, incomprehensible proportions.
