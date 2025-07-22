import { Component } from '@angular/core';
import { Router } from '@angular/router';

@Component({
  selector: 'app-boton-flotante',
  standalone: true,
  imports: [],
  templateUrl: './boton-flotante.component.html',
  styleUrls: ['./boton-flotante.component.css']
})
export class BotonFlotanteComponent {
  buttonLabel = 'Ir al Chat';

  constructor(private router: Router) {
    this.updateButtonLabel();
    this.router.events.subscribe(() => this.updateButtonLabel());
  }

  updateButtonLabel() {
    const currentUrl = this.router.url;
    if (currentUrl === '/chat') {
      this.buttonLabel = 'Ir al Dashboard';
    } else {
      this.buttonLabel = 'Ir al Chat';
    }
  }

  switchPage() {
    if (this.router.url === '/chat') {
      this.router.navigate(['/dashboard']);
    } else {
      this.router.navigate(['/chat']);
    }
  }
}
