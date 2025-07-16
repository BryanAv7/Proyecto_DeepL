import { Component, OnInit } from '@angular/core';
import { SaludoService } from './saludo.service';
import { CommonModule } from '@angular/common';
import { HttpClientModule } from '@angular/common/http';
import { Router, RouterModule } from '@angular/router';
import { AuthService } from './auth.service';
import { Subscription } from 'rxjs';

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [CommonModule, HttpClientModule, RouterModule],
  template: `<router-outlet></router-outlet>`,  // mejor usar router outlet para navegación
})
export class AppComponent implements OnInit {
  mensaje = 'El backend está apagado...';
  private authSub?: Subscription;

  constructor(
    private saludoService: SaludoService,
    private router: Router,
    private authService: AuthService
  ) {}

  ngOnInit() {
    // Escucha el estado de autenticación
    this.authSub = this.authService.authState$.subscribe(user => {
      if (user) {
        // Usuario autenticado, ahora consulta backend
        this.saludoService.obtenerSaludo().subscribe({
          next: (data) => {
            console.log('✅ Backend respondió:', data);
            this.mensaje = data.mensaje;

            if (data?.mensaje) {
              console.log('🚀 Redireccionando al dashboard...');
              this.router.navigate(['/dashboard']);
            }
          },
          error: () => {
            this.mensaje = 'El backend está apagado o no responde.';
            // También puedes decidir qué hacer en caso de error, mostrar mensaje, etc.
          }
        });
      } else {
        // No autenticado, redirige a login
        this.router.navigate(['/login']);
      }
    });
  }

  ngOnDestroy() {
    this.authSub?.unsubscribe();
  }
}
