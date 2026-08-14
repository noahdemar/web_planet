/**
 * Mobile touch controls overlay: on-screen virtual joystick, flight controls,
 * and quick-action buttons for touch devices.
 */

import type { FlyControls } from './controls.js';
import type { Hud } from './hud.js';
import type { TerrainMesh, ShadeMode } from './terrainMesh.js';

export interface MobileControlsCallbacks {
  controls: FlyControls;
  hud: Hud;
  terrain: TerrainMesh;
  onToggleSun: () => void;
  onCycleShadeMode: () => void;
}

export class MobileControls {
  private root: HTMLElement;
  private joystickEl: HTMLElement;
  private joystickThumb: HTMLElement;
  private flyBtnGroup: HTMLElement;
  private walkBtn: HTMLButtonElement;
  private boostBtn: HTMLButtonElement;
  private controls: FlyControls;
  private hud: Hud;

  constructor(private callbacks: MobileControlsCallbacks) {
    this.controls = callbacks.controls;
    this.hud = callbacks.hud;

    this.root = document.createElement('div');
    this.root.id = 'mobile-ui';
    this.root.className = 'mobile-ui';

    // ── Virtual Joystick DOM ─────────────────────────────────────────────────
    const joyContainer = document.createElement('div');
    joyContainer.className = 'touch-joystick';
    joyContainer.style.display = 'none';

    const joyBase = document.createElement('div');
    joyBase.className = 'joystick-base';

    const joyThumb = document.createElement('div');
    joyThumb.className = 'joystick-thumb';

    joyBase.appendChild(joyThumb);
    joyContainer.appendChild(joyBase);
    this.root.appendChild(joyContainer);

    this.joystickEl = joyContainer;
    this.joystickThumb = joyThumb;

    // Connect joystick callback from FlyControls
    this.controls.onJoyChange = (state) => {
      if (state.active) {
        this.joystickEl.style.display = 'block';
        this.joystickEl.style.left = `${state.startX}px`;
        this.joystickEl.style.top = `${state.startY}px`;
        const dx = state.thumbX - state.startX;
        const dy = state.thumbY - state.startY;
        this.joystickThumb.style.transform = `translate(${dx}px, ${dy}px)`;
      } else {
        this.joystickEl.style.display = 'none';
        this.joystickThumb.style.transform = 'translate(0px, 0px)';
      }
    };

    // ── Top Bar Controls (HUD, Sun, Mode) ───────────────────────────────────
    const topBar = document.createElement('div');
    topBar.className = 'mobile-topbar';

    const hudBtn = this.createButton('📊 HUD', 'Toggle debug HUD', () => {
      this.hud.toggle();
    });

    const sunBtn = this.createButton('☀️ Sun', 'Toggle sun cycle / lock', () => {
      callbacks.onToggleSun();
    });

    const modeBtn = this.createButton('🎨 Mode', 'Cycle shading modes', () => {
      callbacks.onCycleShadeMode();
    });

    topBar.appendChild(hudBtn);
    topBar.appendChild(sunBtn);
    topBar.appendChild(modeBtn);
    this.root.appendChild(topBar);

    // ── Bottom Action Cluster ───────────────────────────────────────────────
    const actionCluster = document.createElement('div');
    actionCluster.className = 'mobile-action-cluster';

    // Up / Down flight controls (hidden during walk mode)
    const flyGroup = document.createElement('div');
    flyGroup.className = 'mobile-flight-group';

    const upBtn = this.createHoldButton('▲ Up', 'Ascend in flight mode', (active) => {
      this.controls.verticalInput = active ? 1 : 0;
    });

    const downBtn = this.createHoldButton('▼ Down', 'Descend in flight mode', (active) => {
      this.controls.verticalInput = active ? -1 : 0;
    });

    flyGroup.appendChild(upBtn);
    flyGroup.appendChild(downBtn);
    actionCluster.appendChild(flyGroup);
    this.flyBtnGroup = flyGroup;

    // Boost / Sprint button
    const boostBtn = this.createHoldButton('⚡ Boost', 'Fast travel / Sprint', (active) => {
      this.controls.boostInput = active;
      boostBtn.classList.toggle('active-state', active);
    });
    this.boostBtn = boostBtn;
    actionCluster.appendChild(boostBtn);

    // Walk / Fly toggle button
    const walkBtn = this.createButton('🚶 Walk', 'Toggle Walk / Flight', () => {
      this.controls.setWalk(!this.controls.walk);
      this.syncState();
    });
    this.walkBtn = walkBtn;
    actionCluster.appendChild(walkBtn);

    // Navigation jump buttons
    const navGroup = document.createElement('div');
    navGroup.className = 'mobile-nav-group';

    const landBtn = this.createButton('📍 Land', 'Jump to rugged surface land', () => {
      this.controls.gotoRuggedLand();
      this.syncState();
    });

    const orbitBtn = this.createButton('🪐 Orbit', 'Return to space orbit overview', () => {
      this.controls.reset();
      this.syncState();
    });

    navGroup.appendChild(landBtn);
    navGroup.appendChild(orbitBtn);
    actionCluster.appendChild(navGroup);

    this.root.appendChild(actionCluster);
    document.body.appendChild(this.root);

    this.syncState();
  }

  public syncState(): void {
    if (this.controls.walk) {
      this.walkBtn.textContent = '✈️ Fly';
      this.walkBtn.classList.add('mode-walk');
      this.flyBtnGroup.style.display = 'none';
    } else {
      this.walkBtn.textContent = '🚶 Walk';
      this.walkBtn.classList.remove('mode-walk');
      this.flyBtnGroup.style.display = 'flex';
    }
  }

  private createButton(
    label: string,
    title: string,
    onClick: () => void,
  ): HTMLButtonElement {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'mobile-btn';
    btn.textContent = label;
    btn.title = title;

    const handler = (e: Event) => {
      e.stopPropagation();
      e.preventDefault();
      onClick();
    };

    btn.addEventListener('touchstart', handler, { passive: false });
    btn.addEventListener('click', handler);
    return btn;
  }

  private createHoldButton(
    label: string,
    title: string,
    onStateChange: (active: boolean) => void,
  ): HTMLButtonElement {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'mobile-btn';
    btn.textContent = label;
    btn.title = title;

    const start = (e: Event) => {
      e.stopPropagation();
      e.preventDefault();
      onStateChange(true);
    };

    const end = (e: Event) => {
      e.stopPropagation();
      e.preventDefault();
      onStateChange(false);
    };

    btn.addEventListener('touchstart', start, { passive: false });
    btn.addEventListener('touchend', end, { passive: false });
    btn.addEventListener('touchcancel', end, { passive: false });

    btn.addEventListener('mousedown', start);
    btn.addEventListener('mouseup', end);
    btn.addEventListener('mouseleave', end);

    return btn;
  }
}
